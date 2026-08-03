# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Regenerate the Sortformer diarization golden reference used by the L4/L5 tests.

This produces ``testdata/golden/speech/sortformer_diarization.npz`` by running
the *real* NeMo Sortformer model through the NeMo toolkit (the ground-truth
reference implementation). It must be run inside an environment that has
``nemo_toolkit`` installed (it is **not** a mobius runtime dependency)::

    python -m venv /tmp/nemo_ref_venv
    source /tmp/nemo_ref_venv/bin/activate
    pip install "nemo_toolkit[asr]"
    python scripts/generate_sortformer_golden.py \
        --model nvidia/diar_streaming_sortformer_4spk-v2.1 \
        --revision fafaab5faa1617a0ca52d38dd3dc4bd636800d3d \
        --out testdata/golden/speech/sortformer_diarization.npz

The offline forward path is ``frontend_encoder`` (mel features -> embedding
sequence) followed by ``forward_infer`` (embeddings -> per-frame speaker
activity sigmoids). The committed ``.npz`` stores the mel input, the encoder
embeddings, and the speaker probabilities, plus a ``meta`` JSON blob (model id,
revision, NeMo version, dtype, seed) so the reference is self-describing and
auditable.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

# Deterministic mel-feature fixture (also recorded in metadata).
_SEED = 0
_T = 400  # mel frames; with 8x subsampling -> 50 output diarization frames.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="nvidia/diar_streaming_sortformer_4spk-v2.1")
    parser.add_argument(
        "--revision",
        default="fafaab5faa1617a0ca52d38dd3dc4bd636800d3d",
        help="HuggingFace Hub commit SHA to pin the reference model.",
    )
    parser.add_argument(
        "--out",
        default="testdata/golden/speech/sortformer_diarization.npz",
    )
    args = parser.parse_args()

    import nemo  # type: ignore[import-not-found]
    from huggingface_hub import hf_hub_download
    from nemo.collections.asr.models import (  # type: ignore[import-not-found]
        SortformerEncLabelModel,
    )

    torch.manual_seed(_SEED)

    nemo_path = hf_hub_download(
        repo_id=args.model,
        filename="diar_streaming_sortformer_4spk-v2.1.nemo",
        revision=args.revision,
    )
    model = SortformerEncLabelModel.restore_from(nemo_path, map_location="cpu")
    model.eval()
    # Offline (non-streaming) forward path: full-context attention.
    model.streaming_mode = False

    feat_dim = int(model.cfg.encoder.feat_in)
    mel = torch.randn(1, feat_dim, _T)
    mel_len = torch.tensor([_T], dtype=torch.long)

    with torch.no_grad():
        emb_seq, emb_len = model.frontend_encoder(
            processed_signal=mel, processed_signal_length=mel_len
        )
        preds = model.forward_infer(emb_seq, emb_len)

    num_spks = int(preds.shape[-1])
    meta = {
        "model_id": args.model,
        "revision": args.revision,
        "nemo_version": nemo.__version__,
        "dtype": "float32",
        "seed": _SEED,
        "feat_dim": feat_dim,
        "input_frames": _T,
        "num_spks": num_spks,
    }

    np.savez_compressed(
        args.out,
        mel=mel.numpy().astype(np.float32),
        emb_seq=emb_seq.numpy().astype(np.float32),
        emb_len=emb_len.numpy().astype(np.int64),
        preds=preds.numpy().astype(np.float32),
        meta=np.array(json.dumps(meta)),
    )
    print(f"saved {args.out}\n{json.dumps(meta, indent=2)}")


if __name__ == "__main__":
    main()
