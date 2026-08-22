# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Reusable ONNX policy graphs for hierarchical audio generation workflows."""

from __future__ import annotations

import onnx_ir as ir

from mobius.generation._policy_components import (
    PolicyComponent,
    _component,
    _make_graph,
    _set_public_shape,
)


def build_guided_vocabulary_slice(
    *,
    vocabulary_start: int,
    vocabulary_size: int,
    stop_token_id: int,
    guidance_scale: float,
    conditional_top_k: int,
    dtype: ir.DataType,
) -> PolicyComponent:
    """Restrict two-row decoder logits and apply conditional/unconditional CFG."""
    graph, builder = _make_graph("guided_vocabulary_slice")
    op = builder.op
    logits = builder.input("logits", dtype, [2, "sequence", "vocabulary"])
    last = op.Gather(logits, op.Constant(value_int=-1), axis=1)
    last = op.Cast(last, to=ir.DataType.FLOAT)
    semantic = op.Slice(
        last,
        op.Constant(value_ints=[vocabulary_start]),
        op.Constant(value_ints=[vocabulary_start + vocabulary_size]),
        op.Constant(value_ints=[1]),
    )
    stop = op.Gather(last, op.Constant(value_ints=[stop_token_id]), axis=1)
    stop = op.Unsqueeze(stop, op.Constant(value_ints=[1]))
    candidates = op.Concat(semantic, stop, axis=1)
    conditional = op.Slice(
        candidates,
        op.Constant(value_ints=[0]),
        op.Constant(value_ints=[1]),
        op.Constant(value_ints=[0]),
    )
    unconditional = op.Slice(
        candidates,
        op.Constant(value_ints=[1]),
        op.Constant(value_ints=[2]),
        op.Constant(value_ints=[0]),
    )
    guided = op.Add(
        unconditional,
        op.Mul(
            op.Sub(conditional, unconditional),
            op.Constant(value_float=guidance_scale),
        ),
    )
    top_values, _ = op.TopK(
        conditional,
        op.Constant(value_ints=[conditional_top_k]),
        axis=-1,
        largest=1,
        sorted=1,
        _outputs=2,
    )
    threshold = op.Gather(top_values, op.Constant(value_int=-1), axis=1)
    threshold = op.Unsqueeze(threshold, op.Constant(value_ints=[1]))
    blocked = op.Expand(
        op.Constant(value_float=-3.4028234663852886e38),
        op.Shape(guided),
    )
    guided = op.Where(op.GreaterOrEqual(conditional, threshold), guided, blocked)
    _set_public_shape(guided, [1, vocabulary_size + 1])
    builder.add_output(guided, "candidate_logits")
    return _component("mobius.policy.guided-vocabulary-slice@1", graph, {})


def build_candidate_token_map(
    *,
    vocabulary_start: int,
    vocabulary_size: int,
    stop_token_id: int,
) -> PolicyComponent:
    """Map a sampled compact candidate index back to a vocabulary token."""
    graph, builder = _make_graph("candidate_token_map")
    op = builder.op
    candidate = builder.input("candidate", ir.DataType.INT64, [1])
    is_stop = op.Equal(candidate, op.Constant(value_int=vocabulary_size))
    token = op.Where(
        is_stop,
        op.Constant(value_int=stop_token_id),
        op.Add(candidate, op.Constant(value_int=vocabulary_start)),
    )
    semantic = op.Expand(
        op.Unsqueeze(candidate, op.Constant(value_ints=[1])),
        op.Constant(value_ints=[2, 1]),
    )
    semantic_token = op.Expand(
        op.Unsqueeze(token, op.Constant(value_ints=[1])),
        op.Constant(value_ints=[2, 1]),
    )
    builder.add_output(token, "token")
    builder.add_output(semantic, "semantic_code")
    builder.add_output(semantic_token, "semantic_token")
    _set_public_shape(is_stop, ["batch"])
    builder.add_output(is_stop, "is_stop")
    return _component("mobius.policy.candidate-token-map@1", graph, {})


def build_codebook_embedding_id(*, codebook_size: int) -> PolicyComponent:
    """Map a residual code and one-based codebook induction value to its table row."""
    graph, builder = _make_graph("codebook_embedding_id")
    op = builder.op
    token = builder.input("token", ir.DataType.INT64, [1])
    index = builder.input("codebook_index", ir.DataType.INT64, [])
    offset = op.Mul(
        op.Sub(index, op.Constant(value_int=1)),
        op.Constant(value_int=codebook_size),
    )
    embedding_id = op.Add(token, offset)
    embedding_ids = op.Expand(
        op.Unsqueeze(embedding_id, op.Constant(value_ints=[0])),
        op.Constant(value_ints=[2, 1]),
    )
    builder.add_output(embedding_ids, "embedding_ids")
    return _component("mobius.policy.codebook-embedding-id@1", graph, {})


def build_last_sequence_value(
    dtype: ir.DataType,
    *,
    rows: int | str = "batch",
    channels: int | str = "channels",
) -> PolicyComponent:
    """Select the final sequence item while retaining a length-one axis."""
    graph, builder = _make_graph("last_sequence_value")
    op = builder.op
    value = builder.input("value", dtype, [rows, "sequence", channels])
    last = op.Gather(value, op.Constant(value_int=-1), axis=1)
    last = op.Unsqueeze(last, op.Constant(value_ints=[1]))
    _set_public_shape(last, [rows, 1, channels])
    builder.add_output(last, "last")
    return _component("mobius.policy.last-sequence-value@1", graph, {})


def build_local_rvq_initializer(
    dtype: ir.DataType,
    *,
    hidden_size: int,
) -> PolicyComponent:
    """Create the growing local-decoder sequence and empty per-frame accumulators."""
    graph, builder = _make_graph("local_rvq_initializer")
    op = builder.op
    global_hidden = builder.input("global_hidden", dtype, [2, 1, hidden_size])
    semantic = builder.input("semantic_embedding", dtype, [2, 1, hidden_size])
    sequence = op.Concat(global_hidden, semantic, axis=1)
    empty_codes = op.ConstantOfShape(
        op.Constant(value_ints=[2, 0]),
        value=ir.tensor([0], dtype=ir.DataType.INT64),
    )
    empty_hidden = op.ConstantOfShape(
        op.Constant(value_ints=[1, 0, hidden_size]),
        value=ir.tensor([0.0], dtype=dtype),
    )
    _set_public_shape(sequence, [2, "sequence", hidden_size])
    _set_public_shape(empty_codes, [2, "codes"])
    _set_public_shape(empty_hidden, [1, "parts", hidden_size])
    builder.add_output(sequence, "sequence")
    builder.add_output(empty_codes, "acoustic_codes")
    builder.add_output(empty_hidden, "local_hidden_parts")
    return _component("mobius.policy.local-rvq-initializer@1", graph, {})


def build_local_codebook_select(
    dtype: ir.DataType,
    *,
    guidance_scale: float,
) -> PolicyComponent:
    """Select one local codebook's final-step logits and apply two-row CFG."""
    graph, builder = _make_graph("local_codebook_select")
    op = builder.op
    logits = builder.input(
        "all_codebook_logits",
        dtype,
        ["codebooks", 2, "sequence", "vocabulary"],
    )
    index = builder.input("codebook_index", ir.DataType.INT64, [])
    selected = op.Gather(
        logits,
        op.Sub(index, op.Constant(value_int=1)),
        axis=0,
    )
    selected = op.Gather(selected, op.Constant(value_int=-1), axis=1)
    selected = op.Cast(selected, to=ir.DataType.FLOAT)
    conditional = op.Slice(
        selected,
        op.Constant(value_ints=[0]),
        op.Constant(value_ints=[1]),
        op.Constant(value_ints=[0]),
    )
    unconditional = op.Slice(
        selected,
        op.Constant(value_ints=[1]),
        op.Constant(value_ints=[2]),
        op.Constant(value_ints=[0]),
    )
    guided = op.Add(
        unconditional,
        op.Mul(
            op.Sub(conditional, unconditional),
            op.Constant(value_float=guidance_scale),
        ),
    )
    builder.add_output(guided, "logits")
    return _component("mobius.policy.local-codebook-select@1", graph, {})


def build_local_rvq_append(
    dtype: ir.DataType,
    *,
    hidden_size: int,
) -> PolicyComponent:
    """Append one sampled residual code embedding and its hidden state."""
    graph, builder = _make_graph("local_rvq_append")
    op = builder.op
    sequence = builder.input("sequence", dtype, [2, "sequence", hidden_size])
    projected_embedding = builder.input("projected_embedding", dtype, [2, 1, hidden_size])
    acoustic_codes = builder.input("acoustic_codes", ir.DataType.INT64, [2, "codes"])
    token = builder.input("token", ir.DataType.INT64, [1])
    hidden_states = builder.input("hidden_states", dtype, [2, "sequence", hidden_size])
    repeated = op.Expand(token, op.Constant(value_ints=[2]))
    next_codes = op.Concat(acoustic_codes, op.Unsqueeze(repeated, [1]), axis=1)
    next_sequence = op.Concat(sequence, projected_embedding, axis=1)
    last_hidden = op.Gather(hidden_states, op.Constant(value_int=-1), axis=1)
    conditional_hidden = op.Slice(
        last_hidden,
        op.Constant(value_ints=[0]),
        op.Constant(value_ints=[1]),
        op.Constant(value_ints=[0]),
    )
    hidden_parts = builder.input(
        "local_hidden_parts",
        dtype,
        [1, "parts", hidden_size],
    )
    next_hidden_parts = op.Concat(
        hidden_parts,
        op.Unsqueeze(conditional_hidden, [1]),
        axis=1,
    )
    builder.add_output(next_sequence, "next_sequence")
    builder.add_output(next_codes, "next_acoustic_codes")
    builder.add_output(next_hidden_parts, "next_local_hidden_parts")
    return _component("mobius.policy.local-rvq-append@1", graph, {})


def build_frame_hidden_append(
    dtype: ir.DataType,
    *,
    hidden_size: int,
    num_codebooks: int,
) -> PolicyComponent:
    """Assemble Global plus Local hidden slices and append one acoustic frame."""
    graph, builder = _make_graph("frame_hidden_append")
    op = builder.op
    history = builder.input(
        "history",
        dtype,
        [1, "frames", hidden_size * num_codebooks],
    )
    global_hidden = builder.input("global_hidden", dtype, [2, 1, hidden_size])
    local_parts = builder.input(
        "local_hidden_parts",
        dtype,
        [1, num_codebooks - 1, hidden_size],
    )
    conditional_global = op.Slice(
        global_hidden,
        op.Constant(value_ints=[0]),
        op.Constant(value_ints=[1]),
        op.Constant(value_ints=[0]),
    )
    local_flat = op.Reshape(
        local_parts,
        op.Constant(value_ints=[1, 1, (num_codebooks - 1) * hidden_size]),
    )
    frame = op.Concat(conditional_global, local_flat, axis=2)
    next_history = op.Concat(history, frame, axis=1)
    builder.add_output(next_history, "next_history")
    return _component("mobius.policy.frame-hidden-append@1", graph, {})


def build_embedding_sum(dtype: ir.DataType, *, hidden_size: int) -> PolicyComponent:
    """Add semantic and residual-codebook embeddings for decoder feedback."""
    graph, builder = _make_graph("embedding_sum")
    op = builder.op
    semantic = builder.input("semantic", dtype, [2, 1, hidden_size])
    acoustic = builder.input("acoustic", dtype, [2, 1, hidden_size])
    feedback = op.Add(semantic, acoustic)
    builder.add_output(feedback, "feedback")
    return _component("mobius.policy.embedding-sum@1", graph, {})


def build_acoustic_code_frame(*, num_residual_codebooks: int) -> PolicyComponent:
    """Add the frame axis expected by a complete-codebook feedback embedder."""
    graph, builder = _make_graph("acoustic_code_frame")
    op = builder.op
    acoustic_codes = builder.input(
        "acoustic_codes",
        ir.DataType.INT64,
        [2, num_residual_codebooks],
    )
    framed = op.Unsqueeze(acoustic_codes, [1])
    builder.add_output(framed, "framed_acoustic_codes")
    return _component("mobius.policy.acoustic-code-frame@1", graph, {})


def build_request_continue() -> PolicyComponent:
    """Negate a single-request done flag while retaining request alignment."""
    graph, builder = _make_graph("request_continue")
    done = builder.input("done", ir.DataType.BOOL, ["batch"])
    continued = builder.op.Not(done)
    _set_public_shape(continued, ["batch"])
    builder.add_output(continued, "continue")
    return _component("mobius.policy.request-continue@1", graph, {})


def build_autoregressive_audio_initializer(
    dtype: ir.DataType,
    *,
    fused_hidden_size: int,
) -> PolicyComponent:
    """Initialize explicit frame history and RNG state for one audio request."""
    graph, builder = _make_graph("autoregressive_audio_initializer")
    op = builder.op
    empty_history = op.ConstantOfShape(
        op.Constant(value_ints=[1, 0, fused_hidden_size]),
        value=ir.tensor([0.0], dtype=dtype),
    )
    counter = op.Constant(value_ints=[0])
    active = op.Cast(op.Constant(value_ints=[1]), to=ir.DataType.BOOL)
    done = op.Cast(op.Constant(value_ints=[0]), to=ir.DataType.BOOL)
    _set_public_shape(counter, ["batch"])
    _set_public_shape(active, ["batch"])
    _set_public_shape(done, ["batch"])
    builder.add_output(empty_history, "frame_history")
    builder.add_output(counter, "rng_counter")
    builder.add_output(active, "active")
    builder.add_output(done, "done")
    return _component("mobius.policy.autoregressive-audio-initializer@1", graph, {})


def build_drop_first_frame(dtype: ir.DataType, *, fused_hidden_size: int) -> PolicyComponent:
    """Remove warmup and, when stopped, the terminal non-audio frame."""
    graph, builder = _make_graph("drop_first_frame")
    op = builder.op
    history = builder.input("history", dtype, [1, "frames_with_warmup", fused_hidden_size])
    stopped = builder.input("stopped", ir.DataType.BOOL, [1])
    frame_count = op.Squeeze(op.Shape(history, start=1, end=2), [0])
    emitted_end = op.Sub(
        frame_count,
        op.Squeeze(op.Cast(stopped, to=ir.DataType.INT64), [0]),
    )
    emitted = op.Slice(
        history,
        op.Constant(value_ints=[1]),
        op.Unsqueeze(emitted_end, [0]),
        op.Constant(value_ints=[1]),
    )
    _set_public_shape(emitted, [1, "frames", fused_hidden_size])
    builder.add_output(emitted, "frame_hiddens")
    return _component("mobius.policy.drop-first-frame@1", graph, {})


def build_chunk_plan(
    dtype: ir.DataType,
    *,
    fused_hidden_size: int,
    chunk_frames: int,
    chunk_hop: int,
    latent_channels: int,
    condition_size: int,
) -> PolicyComponent:
    """Compute overlapping frame-window count and initialize chunk carry."""
    graph, builder = _make_graph("audio_chunk_plan")
    op = builder.op
    frame_hiddens = builder.input("frame_hiddens", dtype, [1, "frames", fused_hidden_size])
    frames = op.Squeeze(op.Shape(frame_hiddens, start=1, end=2), [0])
    excess = op.Max(
        op.Sub(frames, op.Constant(value_int=chunk_hop)),
        op.Constant(value_int=1),
    )
    chunk_count = op.Div(
        op.Add(excess, op.Constant(value_int=chunk_hop - 1)),
        op.Constant(value_int=chunk_hop),
    )
    chunk_count = op.Max(chunk_count, op.Constant(value_int=1))
    empty_waveform = op.ConstantOfShape(
        op.Constant(value_ints=[1, 2, 0]),
        value=ir.tensor([0.0], dtype=ir.DataType.FLOAT),
    )
    _set_public_shape(empty_waveform, ["batch", 2, "samples"])
    empty_latent = op.ConstantOfShape(
        op.Constant(value_ints=[1, latent_channels, 0]),
        value=ir.tensor([0.0], dtype=dtype),
    )
    empty_condition = op.ConstantOfShape(
        op.Constant(value_ints=[1, 0, condition_size]),
        value=ir.tensor([0.0], dtype=dtype),
    )
    builder.add_output(chunk_count, "chunk_count")
    builder.add_output(empty_waveform, "waveform")
    builder.add_output(empty_latent, "previous_latent")
    builder.add_output(empty_condition, "previous_condition")
    return _component(
        "mobius.policy.audio-chunk-plan@1",
        graph,
        {"chunk_frames": chunk_frames, "chunk_hop": chunk_hop},
    )


def build_chunk_slice(
    dtype: ir.DataType,
    *,
    fused_hidden_size: int,
    chunk_frames: int,
    chunk_hop: int,
) -> PolicyComponent:
    """Select one 200-frame-style overlapping window from frame hidden states."""
    graph, builder = _make_graph("audio_chunk_slice")
    op = builder.op
    frame_hiddens = builder.input("frame_hiddens", dtype, [1, "frames", fused_hidden_size])
    chunk_index = builder.input("chunk_index", ir.DataType.INT64, [])
    start = op.Mul(chunk_index, op.Constant(value_int=chunk_hop))
    end = op.Min(
        op.Add(start, op.Constant(value_int=chunk_frames)),
        op.Squeeze(op.Shape(frame_hiddens, start=1, end=2), [0]),
    )
    chunk = op.Slice(
        frame_hiddens,
        op.Unsqueeze(start, [0]),
        op.Unsqueeze(end, [0]),
        op.Constant(value_ints=[1]),
    )
    _set_public_shape(chunk, [1, "chunk_frames", fused_hidden_size])
    builder.add_output(chunk, "frame_chunk")
    return _component(
        "mobius.policy.audio-chunk-slice@1",
        graph,
        {"chunk_frames": chunk_frames, "chunk_hop": chunk_hop},
    )


def build_chunk_overlap_prepare(
    dtype: ir.DataType,
    *,
    latent_channels: int,
    condition_size: int,
) -> PolicyComponent:
    """Splice the carried conditioning prefix and describe a latent noise draw."""
    graph, builder = _make_graph("audio_chunk_overlap_prepare")
    op = builder.op
    condition = builder.input("condition", dtype, [1, "latent_length", condition_size])
    previous_condition = builder.input(
        "previous_condition", dtype, [1, "carry_length", condition_size]
    )
    previous_latent = builder.input(
        "previous_latent", dtype, [1, latent_channels, "carry_length"]
    )
    overlap = op.Min(
        op.Squeeze(op.Shape(previous_latent, start=2, end=3), [0]),
        op.Squeeze(op.Shape(condition, start=1, end=2), [0]),
    )
    condition_length = op.Squeeze(op.Shape(condition, start=1, end=2), [0])
    carried = op.Slice(
        previous_condition,
        op.Constant(value_ints=[0]),
        op.Unsqueeze(overlap, [0]),
        op.Constant(value_ints=[1]),
    )
    suffix = op.Slice(
        condition,
        op.Unsqueeze(overlap, [0]),
        op.Unsqueeze(condition_length, [0]),
        op.Constant(value_ints=[1]),
    )
    spliced = op.Concat(carried, suffix, axis=1)
    guided_condition = op.Concat(
        spliced,
        op.ConstantOfShape(op.Shape(spliced), value=ir.tensor([0.0], dtype=dtype)),
        axis=0,
    )
    row_shape = op.Concat(
        op.Constant(value_ints=[latent_channels]),
        op.Shape(condition, start=1, end=2),
        axis=0,
    )
    builder.add_output(guided_condition, "guided_condition")
    builder.add_output(spliced, "spliced_condition")
    builder.add_output(overlap, "overlap")
    builder.add_output(row_shape, "noise_row_shape")
    return _component("mobius.policy.audio-chunk-overlap-prepare@1", graph, {})


def build_overlap_blend(dtype: ir.DataType, *, latent_channels: int) -> PolicyComponent:
    """Blend a chunk's noisy prefix toward the prior latent at flow time ``t``."""
    graph, builder = _make_graph("audio_overlap_blend")
    op = builder.op
    latents = builder.input("latents", dtype, [1, latent_channels, "latent_length"])
    initial_noise = builder.input(
        "initial_noise", dtype, [1, latent_channels, "latent_length"]
    )
    previous_latent = builder.input(
        "previous_latent", dtype, [1, latent_channels, "carry_length"]
    )
    overlap = builder.input("overlap", ir.DataType.INT64, [])
    timestep = builder.input("timestep", dtype, [1])
    previous_prefix = op.Slice(
        previous_latent,
        op.Constant(value_ints=[0]),
        op.Unsqueeze(overlap, [0]),
        op.Constant(value_ints=[2]),
    )
    noise_prompt = op.Slice(
        initial_noise,
        op.Constant(value_ints=[0]),
        op.Unsqueeze(overlap, [0]),
        op.Constant(value_ints=[2]),
    )
    prefix = op.Add(
        op.Mul(
            op.Sub(
                op.CastLike(op.Constant(value_float=1.0), timestep),
                op.Mul(
                    op.CastLike(op.Constant(value_float=0.999999), timestep),
                    timestep,
                ),
            ),
            noise_prompt,
        ),
        op.Mul(timestep, previous_prefix),
    )
    latent_length = op.Squeeze(op.Shape(latents, start=2, end=3), [0])
    suffix = op.Slice(
        latents,
        op.Unsqueeze(overlap, [0]),
        op.Unsqueeze(latent_length, [0]),
        op.Constant(value_ints=[2]),
    )
    blended = op.Concat(prefix, suffix, axis=2)
    builder.add_output(blended, "blended_latents")
    return _component("mobius.policy.audio-overlap-blend@1", graph, {})


def build_flow_guidance(
    dtype: ir.DataType,
    *,
    latent_channels: int,
    guidance_scale: float,
) -> PolicyComponent:
    """Combine conditional row zero and unconditional row one of a 1D velocity."""
    graph, builder = _make_graph("audio_flow_guidance")
    op = builder.op
    sample = builder.input("sample", dtype, [2, latent_channels, "latent_length"])
    conditional = op.Slice(sample, op.Constant(value_ints=[0]), op.Constant(value_ints=[1]))
    unconditional = op.Slice(sample, op.Constant(value_ints=[1]), op.Constant(value_ints=[2]))
    velocity = op.Add(
        unconditional,
        op.Mul(
            op.CastLike(op.Constant(value_float=guidance_scale), sample),
            op.Sub(conditional, unconditional),
        ),
    )
    builder.add_output(velocity, "velocity")
    return _component("mobius.policy.audio-flow-guidance@1", graph, {})


def build_flow_model_inputs(
    dtype: ir.DataType,
    *,
    latent_channels: int,
) -> PolicyComponent:
    """Duplicate one latent and scalar timestep for conditional/unconditional rows."""
    graph, builder = _make_graph("audio_flow_model_inputs")
    op = builder.op
    latents = builder.input("latents", dtype, [1, latent_channels, "latent_length"])
    timestep = builder.input("timestep", dtype, [1])
    guided_latents = op.Concat(latents, latents, axis=0)
    guided_timestep = op.Expand(timestep, op.Constant(value_ints=[2]))
    builder.add_output(guided_latents, "guided_latents")
    builder.add_output(guided_timestep, "guided_timestep")
    return _component("mobius.policy.audio-flow-model-inputs@1", graph, {})


def build_chunk_carry_update(
    dtype: ir.DataType,
    *,
    latent_channels: int,
    condition_size: int,
    carry_length: int,
) -> PolicyComponent:
    """Restore overlap and retain ``[L-2C:L-C]`` latent/condition carry."""
    graph, builder = _make_graph("audio_chunk_carry_update")
    op = builder.op
    latents = builder.input("latents", dtype, [1, latent_channels, "latent_length"])
    previous_latent = builder.input(
        "previous_latent", dtype, [1, latent_channels, "carry_length"]
    )
    condition = builder.input("condition", dtype, [1, "latent_length", condition_size])
    overlap = builder.input("overlap", ir.DataType.INT64, [])
    latent_length = op.Squeeze(op.Shape(latents, start=2, end=3), [0])
    previous_prefix = op.Slice(
        previous_latent,
        op.Constant(value_ints=[0]),
        op.Unsqueeze(overlap, [0]),
        op.Constant(value_ints=[2]),
    )
    suffix = op.Slice(
        latents,
        op.Unsqueeze(overlap, [0]),
        op.Unsqueeze(latent_length, [0]),
        op.Constant(value_ints=[2]),
    )
    restored = op.Concat(previous_prefix, suffix, axis=2)
    carry_start = op.Max(
        op.Constant(value_int=0),
        op.Sub(latent_length, op.Constant(value_int=2 * carry_length)),
    )
    carry_end = op.Max(
        carry_start,
        op.Sub(latent_length, op.Constant(value_int=carry_length)),
    )
    next_latent = op.Slice(
        restored,
        op.Unsqueeze(carry_start, [0]),
        op.Unsqueeze(carry_end, [0]),
        op.Constant(value_ints=[2]),
    )
    next_condition = op.Slice(
        condition,
        op.Unsqueeze(carry_start, [0]),
        op.Unsqueeze(carry_end, [0]),
        op.Constant(value_ints=[1]),
    )
    builder.add_output(restored, "restored_latents")
    builder.add_output(next_latent, "next_previous_latent")
    builder.add_output(next_condition, "next_previous_condition")
    return _component("mobius.policy.audio-chunk-carry-update@1", graph, {})


def build_waveform_stitch(
    *,
    dtype: ir.DataType,
    latent_hop_length: int,
    crop_left_latents: int,
    crop_right_latents: int,
) -> PolicyComponent:
    """Crop decoded overlap in samples and append one stereo waveform chunk."""
    graph, builder = _make_graph("audio_waveform_stitch")
    op = builder.op
    waveform = builder.input("waveform", dtype, ["batch", 2, "chunk_samples"])
    history = builder.input("history", ir.DataType.FLOAT, ["batch", 2, "samples"])
    chunk_index = builder.input("chunk_index", ir.DataType.INT64, [])
    chunk_count = builder.input("chunk_count", ir.DataType.INT64, [])
    left = op.Where(
        op.Equal(chunk_index, op.Constant(value_int=0)),
        op.Constant(value_int=0),
        op.Constant(value_int=crop_left_latents * latent_hop_length),
    )
    right_crop = op.Where(
        op.Equal(chunk_index, op.Sub(chunk_count, op.Constant(value_int=1))),
        op.Constant(value_int=0),
        op.Constant(value_int=crop_right_latents * latent_hop_length),
    )
    samples = op.Squeeze(op.Shape(waveform, start=2, end=3), [0])
    end = op.Sub(samples, right_crop)
    cropped = op.Slice(
        waveform,
        op.Unsqueeze(left, [0]),
        op.Unsqueeze(end, [0]),
        op.Constant(value_ints=[2]),
    )
    cropped = op.Clip(
        op.Cast(cropped, to=ir.DataType.FLOAT),
        op.Constant(value_float=-1.0),
        op.Constant(value_float=1.0),
    )
    next_history = op.Concat(history, cropped, axis=2)
    _set_public_shape(next_history, ["batch", 2, "next_samples"])
    builder.add_output(next_history, "next_history")
    return _component("mobius.policy.audio-waveform-stitch@1", graph, {})
