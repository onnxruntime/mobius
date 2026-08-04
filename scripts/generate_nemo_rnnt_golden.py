# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Regenerate the FastConformer-RNNT golden reference used by the L4 parity test.

This produces ``testdata/golden/speech/nemotron_fastconformer_rnnt.npz`` by
running the *real* NeMo model through the NeMo toolkit (the ground-truth
reference implementation). It must be run inside an environment that has
``nemo_toolkit`` installed (it is **not** a mobius runtime dependency)::

    python -m venv /tmp/nemo_ref_venv
    source /tmp/nemo_ref_venv/bin/activate
    pip install "nemo_toolkit[asr]==2.7.3"
    python scripts/generate_nemo_rnnt_golden.py \
        --model nvidia/nemotron-speech-streaming-en-0.6b \
        --revision 7a9b763e6c5fb103da690219c049fac917aa50b1 \
        --out testdata/golden/speech/nemotron_fastconformer_rnnt.npz

The committed ``.npz`` stores only the arrays needed by the parity test plus a
``meta`` JSON blob (model id, revision, NeMo version, dtype, seed, ids) so the
reference is self-describing and auditable.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

# Deterministic input feature / token fixtures (also recorded in metadata).
_SEED = 0
_T = 131
_FEAT_DIM = 128
_TOKENS = [3, 5, 7, 9]
_SOS_ID = 1025  # rnnt_num_classes (1024) + 1; zero start-of-sequence embedding
_BLANK_ID = 1024


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
        default="testdata/golden/speech/nemotron_fastconformer_rnnt.npz",
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

    feats = torch.randn(1, _FEAT_DIM, _T)
    length = torch.tensor([_T])
    tokens = torch.tensor([_TOKENS])
    tlen = torch.tensor([len(_TOKENS)])

    with torch.no_grad():
        enc_out, _ = model.encoder(audio_signal=feats, length=length)
        # decoder.predict(add_sos=True) prepends a zero SOS vector -> (B, H, U+1)
        pred_out = model.decoder(targets=tokens, target_length=tlen)[0]
        # Bypass fuse_loss_wer by calling the inner joint on (B, T, D) layout.
        joint_out = model.joint.joint(enc_out.transpose(1, 2), pred_out.transpose(1, 2))

    meta = {
        "model_id": args.model,
        "revision": args.revision,
        "nemo_version": nemo.__version__,
        "dtype": "float32",
        "seed": _SEED,
        "feat_dim": _FEAT_DIM,
        "input_frames": _T,
        "tokens": _TOKENS,
        "sos_id": _SOS_ID,
        "blank_id": _BLANK_ID,
    }

    np.savez_compressed(
        args.out,
        feats=feats.numpy().astype(np.float32),
        enc_out=enc_out.numpy().astype(np.float32),
        tokens=tokens.numpy().astype(np.int64),
        pred_out=pred_out.numpy().astype(np.float32),
        joint_out=joint_out.numpy().astype(np.float32),
        meta=np.array(json.dumps(meta)),
    )
    print(f"saved {args.out}\n{json.dumps(meta, indent=2)}")


if __name__ == "__main__":
    main()
