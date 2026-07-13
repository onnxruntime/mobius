# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Regenerate the cache-aware streaming golden for the FastConformer-RNNT test.

This produces ``testdata/golden/speech/nemotron_fastconformer_rnnt_streaming.npz``
by driving the *real* NeMo encoder's streaming ``forward`` (with explicit
``cache_last_channel`` / ``cache_last_time`` / ``cache_last_channel_len`` state)
over two consecutive feature chunks. It must run inside an environment that has
``nemo_toolkit`` installed (not a mobius runtime dependency)::

    python -m venv /tmp/nemo_ref_venv
    source /tmp/nemo_ref_venv/bin/activate
    pip install "nemo_toolkit[asr]==2.7.3"
    python scripts/generate_nemo_rnnt_streaming_golden.py \
        --model nvidia/nemotron-speech-streaming-en-0.6b \
        --revision 7a9b763e6c5fb103da690219c049fac917aa50b1 \
        --out testdata/golden/speech/nemotron_fastconformer_rnnt_streaming.npz

To keep the committed reference small, only the per-chunk feature inputs and the
encoder outputs / lengths / cache-length scalars are stored (not the full
multi-megabyte cache tensors). The streaming parity test validates cache
correctness implicitly by chaining chunk-0's ONNX output caches into chunk-1 and
matching chunk-1's encoder output against this NeMo reference.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

_SEED = 0
_FEAT_DIM = 128
_CHUNK = 120  # feature frames per streaming chunk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="nvidia/nemotron-speech-streaming-en-0.6b")
    parser.add_argument(
        "--revision",
        default="7a9b763e6c5fb103da690219c049fac917aa50b1",
        help="HuggingFace Hub commit SHA to pin the reference model.",
    )
    parser.add_argument(
        "--out",
        default="testdata/golden/speech/nemotron_fastconformer_rnnt_streaming.npz",
    )
    args = parser.parse_args()

    import nemo  # type: ignore[import-not-found]
    import nemo.collections.asr as nemo_asr  # type: ignore[import-not-found]
    from huggingface_hub import hf_hub_download

    torch.manual_seed(_SEED)

    nemo_path = hf_hub_download(
        repo_id=args.model,
        filename="nemotron-speech-streaming-en-0.6b.nemo",
        revision=args.revision,
    )
    model = nemo_asr.models.ASRModel.restore_from(nemo_path, map_location="cpu")
    model.eval()
    enc = model.encoder
    enc.setup_streaming_params()

    def step(feats, ch, ct, cl):
        with torch.no_grad():
            return enc(
                audio_signal=feats,
                length=torch.tensor([_CHUNK]),
                cache_last_channel=ch,
                cache_last_time=ct,
                cache_last_channel_len=cl,
            )

    ch0, ct0, cl0 = enc.get_initial_cache_state(batch_size=1)
    f0 = torch.randn(1, _FEAT_DIM, _CHUNK)
    out0, len0, ch1, ct1, cl1 = step(f0, ch0, ct0, cl0)
    f1 = torch.randn(1, _FEAT_DIM, _CHUNK)
    out1, len1, ch2, ct2, cl2 = step(f1, ch1, ct1, cl1)
    del ch2, ct2  # full out-caches of the second chunk are not part of the golden

    meta = {
        "model_id": args.model,
        "revision": args.revision,
        "nemo_version": nemo.__version__,
        "dtype": "float32",
        "seed": _SEED,
        "feat_dim": _FEAT_DIM,
        "chunk_frames": _CHUNK,
        "last_channel_cache_size": int(enc.streaming_cfg.last_channel_cache_size),
        "drop_extra_pre_encoded": int(enc.streaming_cfg.drop_extra_pre_encoded),
    }

    np.savez_compressed(
        args.out,
        f0=f0.numpy().astype(np.float32),
        f1=f1.numpy().astype(np.float32),
        out0=out0.numpy().astype(np.float32),
        out1=out1.numpy().astype(np.float32),
        len0=len0.numpy().astype(np.int64),
        len1=len1.numpy().astype(np.int64),
        cl1=cl1.numpy().astype(np.int64),
        cl2=cl2.numpy().astype(np.int64),
        meta=np.array(json.dumps(meta)),
    )
    print(f"saved {args.out}\n{json.dumps(meta, indent=2)}")


if __name__ == "__main__":
    main()
