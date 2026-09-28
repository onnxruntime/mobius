# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Nemotron 3 Diarization: streaming Sortformer speaker-diarization model (HF).

Replicates HuggingFace's ``Nemotron3DiarizationForAudioFrameClassification``
(e.g. ``nvidia/Nemotron-3-Diarization``). The exported ONNX graph(s) consume
mel-spectrogram features and produce per-frame speaker-activity probabilities
for up to ``config.num_speakers`` speakers, ordered by their first arrival in
the audio.

Pipeline (matching ``Nemotron3DiarizationAudioModel`` + ``Nemotron3DiarizationModel``
+ ``Nemotron3DiarizationForAudioFrameClassification`` in HuggingFace's
``modeling_nemotron3_diarization.py``):

    input_features [B, feat, T]
      -> transpose                          [B, T, feat]
      -> feature stacking (subsampling)      [B, T/8, feat*8] -> Linear -> hidden
      -> bidirectional RoPE Transformer encoder (pre-LN)
      -> proj (Linear)                       [B, T/8, head_hidden]
      -> sub-pixel upsampler (Conv1d)         [B, T, head_hidden]
      -> classification head (relu-dense-relu-out_proj) -> sigmoid

Two forwards/tasks are exported:

* **Offline** (``diarization`` task, :class:`~mobius.tasks.DiarizationTask`):
  a single full-sequence forward, no chunking or cache state. Matches
  HuggingFace's own offline forward exactly whenever the whole input fits in
  a single chunk (``config.chunk_length`` encoder frames, ~27s of audio at
  the default subsampling factor of 8).
* **Streaming** (``diarization-streaming`` task,
  :class:`~mobius.tasks.DiarizationStreamingTask`): a per-chunk, stateful
  forward. Every call consumes one chunk of audio (plus a few look-ahead
  frames) and the previous call's Arrival-Order Speaker Cache (AOSC) + FIFO
  queue state, and returns this chunk's speaker probabilities plus updated
  cache state. The AOSC/FIFO buffers are fixed-size graph inputs/outputs
  (like a static KV cache) with explicit scalar occupancy counters, so the
  same ONNX graph can be called repeatedly to process an arbitrarily long
  recording — reproducing HuggingFace's ``Nemotron3DiarizationSpeakerCache``
  bookkeeping exactly, top-k score-based compression included.
"""

from __future__ import annotations

import math

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._configs import Nemotron3DiarizationConfig
from mobius.components import (
    Conv1d,
    LayerNorm,
    Linear,
    apply_rotary_pos_emb,
    get_activation,
    initialize_rope,
)

# LayerNorm epsilon used throughout the HuggingFace reference (``nn.LayerNorm``
# defaults), not exposed as a config field.
_LAYER_NORM_EPS = 1e-5


class _FeatureStacking(nn.Module):
    """Stacks ``subsampling_factor`` consecutive spectrogram frames and projects them.

    Matches ``Nemotron3DiarizationFeatureStacking``: zero-pads the time axis
    to a multiple of ``subsampling_factor`` (the padding is a runtime-computed
    length since it depends on the dynamic sequence length), then reshapes
    consecutive frame groups into the feature axis before projecting to
    ``hidden_size``.
    """

    def __init__(self, config: Nemotron3DiarizationConfig):
        super().__init__()
        self._factor = config.subsampling_factor
        self._stacked_size = config.feat_in * config.subsampling_factor
        self.projection = Linear(self._stacked_size, config.hidden_size, bias=False)

    def forward(self, op: OpBuilder, input_features: ir.Value) -> ir.Value:
        # input_features: [B, T, feat_in].
        sequence_length = op.Squeeze(op.Shape(input_features, start=1, end=2), [0])
        factor = op.Constant(value_int=self._factor)
        remainder = op.Mod(sequence_length, factor, fmod=0)
        pad_length = op.Mod(op.Sub(factor, remainder), factor, fmod=0)
        # Pad axis 1 (time) at the end by ``pad_length`` frames.
        pads = op.Concat(
            op.Constant(value_ints=[0, 0, 0, 0]),
            op.Reshape(pad_length, [1]),
            op.Constant(value_ints=[0]),
            axis=0,
        )
        stacked = op.Pad(input_features, pads)
        # [B, T', feat_in] -> [B, T'/factor, feat_in*factor] (row-major
        # reshape, matching PyTorch's contiguous ``.reshape``).
        stacked = op.Reshape(stacked, [0, -1, self._stacked_size])
        return self.projection(op, stacked)


class _AudioAttention(nn.Module):
    """Bidirectional GQA-capable audio attention with full RoPE.

    Matches ``Nemotron3DiarizationAttention``: unbiased Q/K/V projections
    and a biased output projection, no causal masking (single unpadded
    utterance, so no attention bias is needed either).
    """

    def __init__(self, config: Nemotron3DiarizationConfig):
        super().__init__()
        self._num_attention_heads = config.num_attention_heads
        self._num_key_value_heads = config.num_key_value_heads
        self._head_dim = config.head_dim
        prf = config.partial_rotary_factor if config.partial_rotary_factor is not None else 1.0
        self._rotary_dim = 0 if math.isclose(prf, 1.0) else int(self._head_dim * prf)
        self._scale = config.head_dim**-0.5

        q_size = config.num_attention_heads * config.head_dim
        kv_size = config.num_key_value_heads * config.head_dim
        self.q_proj = Linear(config.hidden_size, q_size, bias=False)
        self.k_proj = Linear(config.hidden_size, kv_size, bias=False)
        self.v_proj = Linear(config.hidden_size, kv_size, bias=False)
        self.o_proj = Linear(q_size, config.hidden_size, bias=True)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        position_embeddings: tuple[ir.Value, ir.Value],
    ) -> ir.Value:
        query_states = self.q_proj(op, hidden_states)
        key_states = self.k_proj(op, hidden_states)
        value_states = self.v_proj(op, hidden_states)

        query_states = apply_rotary_pos_emb(
            op,
            query_states,
            position_embeddings,
            num_heads=self._num_attention_heads,
            rotary_embedding_dim=self._rotary_dim,
        )
        key_states = apply_rotary_pos_emb(
            op,
            key_states,
            position_embeddings,
            num_heads=self._num_key_value_heads,
            rotary_embedding_dim=self._rotary_dim,
        )

        attention_output = op.Attention(
            query_states,
            key_states,
            value_states,
            None,
            None,
            None,
            q_num_heads=self._num_attention_heads,
            kv_num_heads=self._num_key_value_heads,
            scale=self._scale,
            is_causal=0,
        )
        return self.o_proj(op, attention_output)


class _AudioMLP(nn.Module):
    """Feed-forward network: ``fc1 -> activation -> fc2`` (both biased)."""

    def __init__(self, config: Nemotron3DiarizationConfig):
        super().__init__()
        self.fc1 = Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = Linear(config.intermediate_size, config.hidden_size, bias=True)
        self._activation = get_activation(config.hidden_act)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        return self.fc2(op, self._activation(op, self.fc1(op, hidden_states)))


class _AudioLayer(nn.Module):
    """Pre-norm bidirectional transformer layer (``Nemotron3DiarizationAudioLayer``)."""

    def __init__(self, config: Nemotron3DiarizationConfig):
        super().__init__()
        self.self_attn = _AudioAttention(config)
        self.layer_norm1 = LayerNorm(config.hidden_size, eps=_LAYER_NORM_EPS)
        self.mlp = _AudioMLP(config)
        self.layer_norm2 = LayerNorm(config.hidden_size, eps=_LAYER_NORM_EPS)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        position_embeddings: tuple[ir.Value, ir.Value],
    ) -> ir.Value:
        residual = hidden_states
        hidden_states = self.layer_norm1(op, hidden_states)
        hidden_states = self.self_attn(op, hidden_states, position_embeddings)
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.layer_norm2(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        return op.Add(residual, hidden_states)


class _AudioTower(nn.Module):
    """RoPE transformer audio encoder (``Nemotron3DiarizationAudioModel``)."""

    def __init__(self, config: Nemotron3DiarizationConfig):
        super().__init__()
        self.embedder = _FeatureStacking(config)
        self.input_layer_norm = LayerNorm(config.hidden_size, eps=_LAYER_NORM_EPS)
        self.layers = nn.ModuleList(
            [_AudioLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.layer_norm = LayerNorm(config.hidden_size, eps=_LAYER_NORM_EPS)
        self.rotary_emb = initialize_rope(config)

    def embed(self, op: OpBuilder, input_features: ir.Value) -> ir.Value:
        """Feature-stacking projection only (raw features -> encoder-frame embeddings).

        Split out from :meth:`encode` so the streaming forward can embed just
        the new chunk's raw features, then concatenate them with cached
        (already-embedded) frames before running the layer stack.
        """
        return self.embedder(op, input_features)

    def encode(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        """Runs the RoPE transformer layer stack over already-embedded frames."""
        sequence_length = op.Squeeze(op.Shape(hidden_states, start=1, end=2), [0])
        position_ids = op.Unsqueeze(
            op.Range(
                op.Constant(value_int=0),
                sequence_length,
                op.Constant(value_int=1),
            ),
            [0],
        )
        if self.rotary_emb is None:
            raise ValueError("Nemotron3Diarization audio encoder requires rotary embeddings")
        position_embeddings = self.rotary_emb(op, position_ids)

        hidden_states = self.input_layer_norm(op, hidden_states)
        for layer in self.layers:
            hidden_states = layer(op, hidden_states, position_embeddings)
        return self.layer_norm(op, hidden_states)

    def forward(self, op: OpBuilder, input_features: ir.Value) -> ir.Value:
        hidden_states = self.embed(op, input_features)
        return self.encode(op, hidden_states)


class _SubpixelUpsampler(nn.Module):
    """Upsamples the encoder frame rate back to the spectrogram frame rate.

    Matches ``Nemotron3DiarizationSubpixelUpsampler``: a Conv1d expands the
    channel axis by ``subsampling_factor``, which is then reshaped into the
    time axis (sub-pixel / depth-to-space upsampling).
    """

    def __init__(self, config: Nemotron3DiarizationConfig):
        super().__init__()
        self._factor = config.subsampling_factor
        self._hidden_size = config.head_hidden_size
        self.conv = Conv1d(
            config.head_hidden_size,
            config.head_hidden_size * config.subsampling_factor,
            kernel_size=3,
            padding=1,
        )

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        # [B, T, C] -> [B, C, T] -> conv -> [B, C*factor, T] -> [B, T, C*factor]
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
        hidden_states = self.conv(op, hidden_states)
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
        # Row-major reshape spreads each time step's C*factor channels across
        # ``factor`` consecutive upsampled time steps of width C.
        return op.Reshape(hidden_states, [0, -1, self._hidden_size])


class _DiarizationBackbone(nn.Module):
    """Audio tower + projection + upsampler (``Nemotron3DiarizationModel``)."""

    def __init__(self, config: Nemotron3DiarizationConfig):
        super().__init__()
        self.audio_tower = _AudioTower(config)
        self.proj = Linear(config.hidden_size, config.head_hidden_size)
        self.upsampler = _SubpixelUpsampler(config)

    def project_upsample(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        hidden_states = self.proj(op, hidden_states)
        return self.upsampler(op, hidden_states)

    def forward(self, op: OpBuilder, input_features: ir.Value) -> ir.Value:
        hidden_states = self.audio_tower(op, input_features)
        return self.project_upsample(op, hidden_states)


class _ClassificationHead(nn.Module):
    """Speaker activity head: ``relu -> dense -> relu -> out_proj``.

    Matches ``Nemotron3DiarizationClassificationHead`` exactly, including
    the (unusual) placement of ReLU *before* both linear layers.
    """

    def __init__(self, config: Nemotron3DiarizationConfig):
        super().__init__()
        self.dense = Linear(config.head_hidden_size, config.head_hidden_size)
        self.out_proj = Linear(config.head_hidden_size, config.num_speakers)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        hidden_states = self.dense(op, op.Relu(hidden_states))
        return self.out_proj(op, op.Relu(hidden_states))


def _scalar(op: OpBuilder, value: ir.Value) -> ir.Value:
    """Reshapes a scalar (rank-0) value to rank-1 shape ``[1]`` for Slice bounds."""
    return op.Reshape(value, op.Constant(value_ints=[1]))


def _pad_time_axis(op: OpBuilder, x: ir.Value, target_length: ir.Value) -> ir.Value:
    """Zero-pads a ``[B, F, C]`` tensor at the end of axis 1 up to ``target_length``.

    ``target_length`` is a scalar (rank-0) ``int64`` value; ``F`` (the current
    length) may be dynamic and is assumed ``<= target_length``.
    """
    current_length = op.Shape(x, start=1, end=2)
    pad_amount = op.Sub(_scalar(op, target_length), current_length)
    zero = op.Constant(value_ints=[0])
    pads = op.Concat(zero, zero, zero, zero, pad_amount, zero, axis=0)
    return op.Pad(x, pads, mode="constant")


def _avg_pool_probs(op: OpBuilder, logits: ir.Value, factor: int) -> ir.Value:
    """Sigmoid + average-pool speaker logits down to the encoder frame rate.

    Matches ``Nemotron3DiarizationSpeakerCache._pool_probs`` (without the
    optional padding mask, which streaming export does not support — see the
    module docstring's streaming limitations).
    """
    probs = op.Sigmoid(logits)
    probs = op.Transpose(probs, perm=[0, 2, 1])
    probs = op.AveragePool(probs, kernel_shape=[factor], strides=[factor])
    return op.Transpose(probs, perm=[0, 2, 1])


def _num_popped_frames(
    op: OpBuilder, num_fifo_total: ir.Value, fifo_capacity: int, update_period: int
) -> ir.Value:
    """Matches ``Nemotron3DiarizationSpeakerCache._num_popped_frames``."""
    over_capacity = op.Sub(num_fifo_total, op.Constant(value_int=fifo_capacity))
    popped = op.Max(op.Constant(value_int=update_period), over_capacity)
    popped = op.Min(popped, num_fifo_total)
    zero = op.Constant(value_int=0)
    has_overflow = op.Greater(num_fifo_total, op.Constant(value_int=fifo_capacity))
    return op.Where(has_overflow, popped, zero)


def _get_frame_scores(
    op: OpBuilder, probs: ir.Value, threshold: float, min_positive_scores: int
) -> ir.Value:
    """Matches ``Nemotron3DiarizationSpeakerCache._get_frame_scores``.

    ``probs``: ``[B, F, S]``. Frames that are all-zero (this module's padding
    convention for "no such frame") are never speech (``probs <= 0.5``), so
    they always resolve to ``-inf`` and are naturally excluded from top-k
    selection without extra masking.

    All float literals below are cast to ``probs``'s dtype: this scoring path
    runs entirely in ``config.dtype`` (fp16/bf16 exports included), and ONNX
    elementwise ops require both operands to share a dtype — a bare FLOAT32
    ``Constant`` combined with an fp16/bf16 tensor is an invalid graph.
    """

    def _lit(value: float) -> ir.Value:
        return op.CastLike(op.Constant(value_float=value), probs)

    threshold_t = _lit(threshold)
    log_probs = op.Log(op.Clip(probs, threshold_t))
    complements = op.Sub(_lit(1.0), probs)
    log_complements = op.Log(op.Clip(complements, threshold_t))
    sum_log_complements = op.ReduceSum(log_complements, [-1], keepdims=1)
    log_half = _lit(math.log(0.5))
    scores = op.Sub(op.Add(op.Sub(log_probs, log_complements), sum_log_complements), log_half)

    neg_inf = _lit(float("-inf"))
    is_speech = op.Greater(probs, _lit(0.5))
    scores = op.Where(is_speech, scores, neg_inf)

    is_positive = op.Greater(scores, _lit(0.0))
    positive_count = op.ReduceSum(op.Cast(is_positive, to=ir.DataType.INT64), [1], keepdims=1)
    has_enough_positive = op.GreaterOrEqual(
        positive_count, op.Constant(value_int=min_positive_scores)
    )
    extra_masked = op.And(op.And(op.Not(is_positive), is_speech), has_enough_positive)
    return op.Where(extra_masked, neg_inf, scores)


def _boost_scores(op: OpBuilder, scores: ir.Value, num_boosted: int, boost: float) -> ir.Value:
    """Matches ``Nemotron3DiarizationSpeakerCache._boost_scores`` (no-op if ``num_boosted <= 0``)."""
    if num_boosted <= 0:
        return scores
    _, indices = op.TopK(
        scores,
        op.Constant(value_ints=[num_boosted]),
        axis=1,
        largest=1,
        sorted=0,
        _outputs=2,
    )
    updates = op.Expand(op.Constant(value_float=boost), op.Shape(indices))
    updates = op.CastLike(updates, scores)
    return op.ScatterElements(scores, indices, updates, axis=1, reduction="add")


def _sort_ascending(op: OpBuilder, values: ir.Value, length: int) -> ir.Value:
    """Sorts a ``[B, length]`` int64 tensor ascending along axis 1 (no ``Sort`` op in ONNX).

    Uses the "negate + TopK(largest, sorted, k=length)" trick: requesting the
    full length back in sorted-descending order of the negation is exactly
    ascending order of the original values.
    """
    negated = op.Cast(op.Neg(values), to=ir.DataType.FLOAT)
    sorted_negated, _ = op.TopK(
        negated,
        op.Constant(value_ints=[length]),
        axis=1,
        largest=1,
        sorted=1,
        _outputs=2,
    )
    return op.Cast(op.Neg(sorted_negated), to=ir.DataType.INT64)


def _compress_speaker_cache(
    op: OpBuilder,
    embeds: ir.Value,
    probs: ir.Value,
    silence_embeds: ir.Value,
    config: Nemotron3DiarizationConfig,
) -> tuple[ir.Value, ir.Value]:
    """Matches ``Nemotron3DiarizationSpeakerCache._compress``.

    ``embeds``: ``[B, F, H]``, ``probs``: ``[B, F, S]`` with ``F`` (dynamic)
    strictly greater than ``speaker_cache_length`` (the caller only invokes
    this when compression is actually needed). Returns exactly
    ``speaker_cache_length`` frames, keyed by score and grouped by speaker
    (highest-scoring frames per speaker, in original temporal order).
    """
    num_speakers = config.num_speakers
    cache_length = config.streaming_speaker_cache_length
    num_silence_frames = config.streaming_silence_frames_per_speaker
    threshold = config.streaming_prediction_score_threshold
    # Per-speaker budget the score policy spends on boosting/positivity,
    # excluding the reserved silence slots (mirrors the Python constants
    # HF precomputes once in ``Nemotron3DiarizationSpeakerCache.__init__``).
    budget = cache_length // num_speakers - num_silence_frames
    min_positive_scores = math.floor(budget * config.streaming_min_positive_scores_rate)
    num_strong_boosted = math.floor(budget * config.streaming_strong_boost_rate)
    num_weak_boosted = math.floor(budget * config.streaming_weak_boost_rate)

    scores = _get_frame_scores(op, probs, threshold, min_positive_scores)

    num_frames = op.Squeeze(op.Shape(embeds, start=1, end=2), [0])
    frame_positions = op.Range(op.Constant(value_int=0), num_frames, op.Constant(value_int=1))
    is_tail = op.GreaterOrEqual(frame_positions, op.Constant(value_int=cache_length))
    tail_boost = op.Where(
        is_tail,
        op.Constant(value_float=config.streaming_latest_frames_score_boost),
        op.Constant(value_float=0.0),
    )
    # ``tail_boost`` is built from bare FLOAT32 literals (matching each
    # other); cast once here to ``scores``'s dtype before combining, rather
    # than casting each literal individually.
    tail_boost = op.CastLike(tail_boost, scores)
    scores = op.Add(scores, op.Unsqueeze(tail_boost, [0, 2]))

    scores = _boost_scores(op, scores, num_strong_boosted, -2.0 * math.log(0.5))
    scores = _boost_scores(op, scores, num_weak_boosted, -math.log(0.5))

    # Append the shared silence embedding/prob row, and reserve
    # ``num_silence_frames`` score slots per speaker with +inf (always kept).
    batch = op.Shape(embeds, start=0, end=1)
    hidden_size = op.Shape(embeds, start=2, end=3)
    silence_row = op.Expand(
        op.Reshape(silence_embeds, op.Constant(value_ints=[1, 1, -1])),
        op.Concat(batch, op.Constant(value_ints=[1]), hidden_size, axis=0),
    )
    embeds = op.Concat(embeds, silence_row, axis=1)
    probs_pad = op.Concat(
        op.Constant(value_ints=[0, 0, 0]), op.Constant(value_ints=[0, 1, 0]), axis=0
    )
    probs = op.Pad(probs, probs_pad, mode="constant")
    scores_pad = op.Concat(
        op.Constant(value_ints=[0, 0, 0]),
        op.Constant(value_ints=[0, num_silence_frames, 0]),
        axis=0,
    )
    scores = op.Pad(
        scores,
        scores_pad,
        op.CastLike(op.Constant(value_float=float("inf")), scores),
        mode="constant",
    )

    num_scored_frames = op.Add(num_frames, op.Constant(value_int=num_silence_frames))
    sentinel = op.Mul(num_scored_frames, op.Constant(value_int=num_speakers))
    flat_scores = op.Reshape(
        op.Transpose(scores, perm=[0, 2, 1]),
        op.Concat(batch, op.Constant(value_ints=[-1]), axis=0),
    )
    topk_scores, topk_indices = op.TopK(
        flat_scores,
        op.Constant(value_ints=[cache_length]),
        axis=1,
        largest=1,
        sorted=0,
        _outputs=2,
    )
    is_masked = op.Equal(
        topk_scores, op.CastLike(op.Constant(value_float=float("-inf")), topk_scores)
    )
    topk_indices = op.Where(is_masked, op.Unsqueeze(sentinel, [0]), topk_indices)
    topk_indices = _sort_ascending(op, topk_indices, cache_length)

    frame_indices = op.Mod(topk_indices, op.Unsqueeze(num_scored_frames, [0]), fmod=0)
    frame_indices = op.Min(frame_indices, op.Unsqueeze(num_frames, [0]))
    is_sentinel = op.Equal(topk_indices, op.Unsqueeze(sentinel, [0]))
    frame_indices = op.Where(is_sentinel, op.Unsqueeze(num_frames, [0]), frame_indices)

    gather_indices = op.Unsqueeze(frame_indices, [-1])
    new_embeds = op.GatherND(embeds, gather_indices, batch_dims=1)
    new_probs = op.GatherND(probs, gather_indices, batch_dims=1)
    return new_embeds, new_probs


class Nemotron3DiarizationModel(nn.Module):
    """Streaming Sortformer speaker-diarization model (HuggingFace port).

    Replicates HuggingFace's ``Nemotron3DiarizationForAudioFrameClassification``
    (e.g. ``nvidia/Nemotron-3-Diarization``): a bidirectional, partial-RoPE
    Transformer audio encoder followed by a sub-pixel upsampler and a speaker
    sigmoid head. Consumes mel-spectrogram features and returns per-frame
    speaker-activity probabilities in ``[0, 1]``.

    Two forwards are exported (see ``tasks/_diarization.py`` and
    ``tasks/_diarization_streaming.py``):

    * :meth:`forward` — the **offline** single full-sequence pass (``diarization``
      task). Matches HuggingFace exactly whenever the whole input fits within
      one chunk (``config.chunk_length`` encoder frames, ~27s of audio).
    * :meth:`forward_streaming` — the **streaming**, per-chunk pass
      (``diarization-streaming`` task): consumes one chunk of audio plus a
      few look-ahead frames, together with the previous step's Arrival-Order
      Speaker Cache (AOSC) + FIFO queue state (fixed-size buffers and scalar
      occupancy counters, all graph inputs/outputs), and returns this chunk's
      speaker probabilities plus the updated cache state. Repeated calls
      reproduce HuggingFace's streaming ``Nemotron3DiarizationSpeakerCache``
      bookkeeping exactly, including its top-k score-based compression.

      Limitation: unlike HuggingFace, the streaming export does not accept a
      padding ``attention_mask`` — every step's input window is assumed fully
      valid (no silence padding within a chunk). This holds for all but
      possibly the very last chunk of a recording, matching the precision
      needed for real-time streaming use.
    """

    default_task: str = "diarization"
    category: str = "Speech-to-Text"
    config_class = Nemotron3DiarizationConfig

    def __init__(self, config: Nemotron3DiarizationConfig):
        super().__init__()
        self.config = config
        self.model = _DiarizationBackbone(config)
        self.classifier = _ClassificationHead(config)
        # Learned embedding filling reserved silence slots when the AOSC is
        # compressed. Unused (and pruned by the optimizer) by the offline,
        # non-streaming graph.
        self.silence_embeds = nn.Parameter([config.hidden_size])

    def forward(self, op: OpBuilder, input_features: ir.Value) -> ir.Value:
        # input_features: [B, feat_in, T] -> [B, T, feat_in], matching the
        # shared ``DiarizationTask`` input contract (also used by sortformer).
        input_features = op.Transpose(input_features, perm=[0, 2, 1])
        # Original (pre-padding) frame count, to trim the upsampled output.
        num_frames = op.Shape(input_features, start=1, end=2)

        hidden_states = self.model(op, input_features)
        logits = self.classifier(op, hidden_states)
        logits = op.Slice(
            logits,
            op.Constant(value_ints=[0]),
            num_frames,
            op.Constant(value_ints=[1]),
        )
        return op.Sigmoid(logits)

    def _call_backbone_scoped(self, op: OpBuilder, method_name: str, *args):
        """Call a bound method on ``self.model``/``self.model.audio_tower`` under proper module scope.

        ``forward_streaming`` calls ``embed``/``encode``/``project_upsample``
        directly (they are not full ``forward`` passes), bypassing
        ``Module.__call__``'s automatic name-qualification and parameter
        realization. Mirrors that qualification manually (see
        ``components/_moe.py``'s ``_realize_gate_and_get_qmoe_routing`` for
        the same pattern) so weight names match the offline export exactly.
        """
        backbone = self.model
        audio_tower = backbone.audio_tower
        builder = op.builder
        builder.push_module(backbone.name or "model", type(backbone).__qualname__)
        try:
            for param in backbone.parameters(recurse=False):
                param._realize(builder)  # pylint: disable=protected-access
            if method_name == "project_upsample":
                return backbone.project_upsample(op, *args)
            builder.push_module(
                audio_tower.name or "audio_tower", type(audio_tower).__qualname__
            )
            try:
                for param in audio_tower.parameters(recurse=False):
                    param._realize(builder)  # pylint: disable=protected-access
                return getattr(audio_tower, method_name)(op, *args)
            finally:
                builder.pop_module()
        finally:
            builder.pop_module()

    def forward_streaming(
        self,
        op: OpBuilder,
        input_features: ir.Value,
        num_lookahead_frames: ir.Value,
        past_cache_embeds: ir.Value,
        past_cache_probs: ir.Value,
        past_fifo: ir.Value,
        past_num_cache_frames: ir.Value,
        past_num_fifo_frames: ir.Value,
        past_is_compressed: ir.Value,
    ) -> tuple[ir.Value, ir.Value, ir.Value, ir.Value, ir.Value, ir.Value, ir.Value]:
        """One streaming chunk step, matching HuggingFace's chunked forward loop body.

        See ``Nemotron3DiarizationForAudioFrameClassification.forward``'s
        per-chunk logic and ``Nemotron3DiarizationSpeakerCache.update``. Every
        cache tensor is a fixed-size buffer (``streaming_speaker_cache_length``/
        ``streaming_fifo_length`` capacity); ``past_num_cache_frames``/
        ``past_num_fifo_frames`` track how much of each buffer is valid, the
        same convention as a static KV cache with an explicit occupancy count.

        Returns ``(speaker_probs, cache_embeds, cache_probs, fifo,
        num_cache_frames, num_fifo_frames, is_compressed)``.
        """
        config = self.config
        factor = config.subsampling_factor
        cache_length = config.streaming_speaker_cache_length
        fifo_capacity = config.streaming_fifo_length
        update_period = config.streaming_speaker_cache_update_period

        # ``forward_streaming`` is invoked directly (not via ``self(op, ...)``),
        # bypassing ``Module.__call__``'s automatic parameter realization, so
        # ``silence_embeds`` (only referenced deep inside the compress-branch
        # subgraph) must be registered as a graph initializer explicitly here.
        self.silence_embeds._realize(op.builder)  # type: ignore[attr-defined]

        # input_features: [B, feat_in, W] -> [B, W, feat_in].
        input_features = op.Transpose(input_features, perm=[0, 2, 1])
        raw_num_frames = op.Shape(input_features, start=1, end=2)

        chunk_embeds = self._call_backbone_scoped(op, "embed", input_features)
        num_new_embeds = op.Squeeze(op.Shape(chunk_embeds, start=1, end=2), [0])
        num_chunk_frames = op.Sub(num_new_embeds, num_lookahead_frames)

        cached_length = op.Add(past_num_cache_frames, past_num_fifo_frames)
        cached_embeds = op.Concat(
            op.Slice(
                past_cache_embeds,
                op.Constant(value_ints=[0]),
                _scalar(op, past_num_cache_frames),
                op.Constant(value_ints=[1]),
            ),
            op.Slice(
                past_fifo,
                op.Constant(value_ints=[0]),
                _scalar(op, past_num_fifo_frames),
                op.Constant(value_ints=[1]),
            ),
            axis=1,
        )
        chunk_input_embeds = op.Concat(cached_embeds, chunk_embeds, axis=1)

        encoded = self._call_backbone_scoped(op, "encode", chunk_input_embeds)
        upsampled = self._call_backbone_scoped(op, "project_upsample", encoded)
        chunk_logits = self.classifier(op, upsampled)

        # This step's speaker-probability output: the chunk region only
        # (excludes both the cached prefix and the look-ahead suffix),
        # trimmed to this call's raw (pre-feature-stacking-padding) length.
        start_logit_idx = op.Mul(cached_length, op.Constant(value_int=factor))
        end_logit_idx = op.Mul(
            op.Add(cached_length, num_chunk_frames), op.Constant(value_int=factor)
        )
        chunk_region = op.Slice(
            chunk_logits,
            _scalar(op, start_logit_idx),
            _scalar(op, end_logit_idx),
            op.Constant(value_ints=[1]),
        )
        chunk_region = op.Slice(
            chunk_region,
            op.Constant(value_ints=[0]),
            raw_num_frames,
            op.Constant(value_ints=[1]),
        )
        speaker_probs = op.Sigmoid(chunk_region)

        # --- Arrival-Order Speaker Cache (AOSC) + FIFO update ---
        probs = _avg_pool_probs(op, chunk_logits, factor)

        new_chunk_embeds = op.Slice(
            chunk_input_embeds,
            _scalar(op, cached_length),
            _scalar(op, op.Add(cached_length, num_chunk_frames)),
            op.Constant(value_ints=[1]),
        )
        old_fifo = op.Slice(
            past_fifo,
            op.Constant(value_ints=[0]),
            _scalar(op, past_num_fifo_frames),
            op.Constant(value_ints=[1]),
        )
        fifo_embeds = op.Concat(old_fifo, new_chunk_embeds, axis=1)
        num_fifo_total = op.Add(past_num_fifo_frames, num_chunk_frames)
        num_popped = _num_popped_frames(op, num_fifo_total, fifo_capacity, update_period)

        fifo_probs_all = op.Slice(
            probs,
            _scalar(op, past_num_cache_frames),
            _scalar(op, op.Add(past_num_cache_frames, num_fifo_total)),
            op.Constant(value_ints=[1]),
        )
        stored_probs_uncompressed = op.Slice(
            probs,
            op.Constant(value_ints=[0]),
            _scalar(op, past_num_cache_frames),
            op.Constant(value_ints=[1]),
        )
        stored_probs_compressed = op.Slice(
            past_cache_probs,
            op.Constant(value_ints=[0]),
            _scalar(op, past_num_cache_frames),
            op.Constant(value_ints=[1]),
        )
        stored_probs = op.Where(
            op.Reshape(past_is_compressed, op.Constant(value_ints=[1, 1, 1])),
            stored_probs_compressed,
            stored_probs_uncompressed,
        )

        popped_embeds = op.Slice(
            fifo_embeds,
            op.Constant(value_ints=[0]),
            _scalar(op, num_popped),
            op.Constant(value_ints=[1]),
        )
        popped_probs = op.Slice(
            fifo_probs_all,
            op.Constant(value_ints=[0]),
            _scalar(op, num_popped),
            op.Constant(value_ints=[1]),
        )
        old_cache_embeds = op.Slice(
            past_cache_embeds,
            op.Constant(value_ints=[0]),
            _scalar(op, past_num_cache_frames),
            op.Constant(value_ints=[1]),
        )
        cache_embeds_candidate = op.Concat(old_cache_embeds, popped_embeds, axis=1)
        cache_probs_candidate = op.Concat(stored_probs, popped_probs, axis=1)
        combined_count = op.Add(past_num_cache_frames, num_popped)
        needs_compress = op.Greater(combined_count, op.Constant(value_int=cache_length))

        def _compress_branch(branch_op: OpBuilder):
            embeds_out, probs_out = _compress_speaker_cache(
                branch_op,
                cache_embeds_candidate,
                cache_probs_candidate,
                self.silence_embeds,
                config,
            )
            count_out = branch_op.Constant(value_int=cache_length)
            return embeds_out, probs_out, count_out

        def _passthrough_branch(branch_op: OpBuilder):
            embeds_out = _pad_time_axis(
                branch_op, cache_embeds_candidate, branch_op.Constant(value_int=cache_length)
            )
            probs_out = _pad_time_axis(
                branch_op, cache_probs_candidate, branch_op.Constant(value_int=cache_length)
            )
            # Recompute (rather than pass through) `combined_count`: a bare
            # outer-scope value cannot be a subgraph output (and any
            # `Identity` wrapper added purely for that purpose is eliminated
            # by mobius's cleanup optimization pass, which also recurses into
            # subgraphs), so this must be a genuine op inside the branch.
            count_out = branch_op.Add(past_num_cache_frames, num_popped)
            return embeds_out, probs_out, count_out

        then_branch = op.builder.subgraph(
            _compress_branch,
            inputs=[],
            outputs=[
                ir.Value(name="compressed_cache_embeds"),
                ir.Value(name="compressed_cache_probs"),
                ir.Value(name="compressed_num_cache_frames"),
            ],
            name="compress_speaker_cache",
        )
        else_branch = op.builder.subgraph(
            _passthrough_branch,
            inputs=[],
            outputs=[
                ir.Value(name="uncompressed_cache_embeds"),
                ir.Value(name="uncompressed_cache_probs"),
                ir.Value(name="uncompressed_num_cache_frames"),
            ],
            name="passthrough_speaker_cache",
        )
        new_cache_embeds, new_cache_probs, new_num_cache_frames = op.If(
            needs_compress, then_branch=then_branch, else_branch=else_branch, _outputs=3
        )
        new_is_compressed = op.Or(past_is_compressed, needs_compress)

        new_fifo_content = op.Slice(
            fifo_embeds,
            _scalar(op, num_popped),
            _scalar(op, num_fifo_total),
            op.Constant(value_ints=[1]),
        )
        new_num_fifo_frames = op.Sub(num_fifo_total, num_popped)
        new_fifo = _pad_time_axis(op, new_fifo_content, op.Constant(value_int=fifo_capacity))

        return (
            speaker_probs,
            new_cache_embeds,
            new_cache_probs,
            new_fifo,
            new_num_cache_frames,
            new_num_fifo_frames,
            new_is_compressed,
        )
