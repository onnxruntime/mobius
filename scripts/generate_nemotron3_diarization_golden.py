#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

r"""Regenerate the Nemotron 3 Diarization golden reference used by the L4/L5 tests.

This produces ``testdata/golden/speech/nemotron3_diarization.npz`` by running
the *real* HuggingFace ``Nemotron3DiarizationForAudioFrameClassification``
checkpoint (the ground-truth reference implementation) on deterministic
synthetic mel input. It must be run inside an environment with a Transformers
build that supports ``nemotron3_diarization`` (merged upstream after the
5.17.0 PyPI release — install with
``pip install "git+https://github.com/huggingface/transformers.git"`` until a
release containing it ships)::

    python scripts/generate_nemotron3_diarization_golden.py \\
        --model nvidia/Nemotron-3-Diarization \\
        --revision f667ed73aee57d40cc39428eb768b4fd87a0a29e \\
        --out testdata/golden/speech/nemotron3_diarization.npz

Two reference passes are recorded:

* **Offline** (single call, no cache): mel length is kept below
  ``config.chunk_length * subsampling_factor`` so HuggingFace's own offline
  forward does not internally re-chunk — this is the exact regime the mobius
  ``DiarizationTask`` (offline) graph is documented to match bit-for-bit.
* **Streaming** (3 chunks): each chunk is a full ``config.chunk_length +
  config.chunk_right_context`` window (matching the *fixed* input shape of
  the mobius ``DiarizationStreamingTask`` ONNX graph), fed through
  HuggingFace's real per-chunk ``forward`` + ``Nemotron3DiarizationSpeakerCache``
  bookkeeping. With the checkpoint's default streaming config (FIFO capacity
  264, speaker-cache capacity 264, update period 222) a single 340-frame
  chunk already overflows the FIFO, so this exercises both the FIFO-to-cache
  eviction *and* the top-k AOSC compression path by the second chunk.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

# Deterministic synthetic mel fixtures (also recorded in metadata).
_SEED = 0
_OFFLINE_T = 2000  # mel frames; well below chunk_length(340) * subsampling(8) = 2720.
_NUM_STREAM_CHUNKS = 3


def _run_offline(model, mel_dim: int) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(_SEED)
    mel = torch.randn(1, _OFFLINE_T, mel_dim)
    with torch.no_grad():
        out = model(input_features=mel)
    return mel.numpy(), out.logits.sigmoid().numpy()


def _run_streaming(model, mel_dim: int) -> dict[str, np.ndarray]:
    chunk_length = model.config.chunk_length
    chunk_right_context = model.config.chunk_right_context
    subsampling = model.config.audio_config.subsampling_factor
    window_encoder = chunk_length + chunk_right_context
    raw_window = window_encoder * subsampling

    arrays: dict[str, np.ndarray] = {}
    cache = None
    for i in range(_NUM_STREAM_CHUNKS):
        torch.manual_seed(_SEED + 100 + i)
        mel = torch.randn(1, raw_window, mel_dim)
        is_last = i == _NUM_STREAM_CHUNKS - 1
        lookahead = 0 if is_last else chunk_right_context
        with torch.no_grad():
            out = model(
                input_features=mel, speaker_cache=cache, num_lookahead_frames=lookahead
            )
        cache = out.speaker_cache
        arrays[f"stream_mel_{i}"] = mel.numpy().copy()
        arrays[f"stream_lookahead_{i}"] = np.array(lookahead, dtype=np.int64)
        arrays[f"stream_preds_{i}"] = out.logits.sigmoid().numpy().copy()
        # NOTE: cache.embeds/probs/fifo are mutable buffers reused (and
        # mutated in place via index_copy_) across chunks, so every snapshot
        # here MUST be copied -- otherwise all per-chunk arrays end up as
        # views onto the same storage and silently alias the *last* chunk's
        # values once np.savez serializes them.
        arrays[f"stream_cache_embeds_{i}"] = cache.embeds.numpy().copy()
        arrays[f"stream_cache_probs_{i}"] = cache.probs.numpy().copy()
        arrays[f"stream_fifo_{i}"] = cache.fifo.numpy().copy()
        arrays[f"stream_num_cache_frames_{i}"] = np.array(
            cache.num_cache_frames, dtype=np.int64
        )
        arrays[f"stream_num_fifo_frames_{i}"] = np.array(cache.num_fifo_frames, dtype=np.int64)
        arrays[f"stream_is_compressed_{i}"] = np.array(cache.is_compressed, dtype=np.bool_)
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="nvidia/Nemotron-3-Diarization")
    parser.add_argument(
        "--revision",
        default="f667ed73aee57d40cc39428eb768b4fd87a0a29e",
        help="HuggingFace Hub commit SHA to pin the reference model.",
    )
    parser.add_argument(
        "--out",
        default="testdata/golden/speech/nemotron3_diarization.npz",
    )
    args = parser.parse_args()

    import transformers
    from transformers import AutoModelForAudioFrameClassification

    torch.manual_seed(_SEED)

    model = AutoModelForAudioFrameClassification.from_pretrained(
        args.model, revision=args.revision, dtype=torch.float32
    )
    model.eval()

    mel_dim = model.config.audio_config.num_mel_bins
    offline_mel, offline_preds = _run_offline(model, mel_dim)
    stream_arrays = _run_streaming(model, mel_dim)

    meta = {
        "model_id": args.model,
        "revision": args.revision,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "seed": _SEED,
        "num_speakers": int(model.config.head_config.num_speakers),
        "mel_dim": mel_dim,
        "chunk_length": int(model.config.chunk_length),
        "chunk_right_context": int(model.config.chunk_right_context),
        "subsampling_factor": int(model.config.audio_config.subsampling_factor),
        "streaming_fifo_length": int(model.config.streaming_config.fifo_length),
        "streaming_speaker_cache_length": int(
            model.config.streaming_config.speaker_cache_length
        ),
        "streaming_speaker_cache_update_period": int(
            model.config.streaming_config.speaker_cache_update_period
        ),
        "num_stream_chunks": _NUM_STREAM_CHUNKS,
    }

    np.savez(
        args.out,
        offline_mel=offline_mel,
        offline_preds=offline_preds,
        meta=json.dumps(meta),
        **stream_arrays,  # type: ignore[arg-type]  # numpy stub mis-resolves the splat here.
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
