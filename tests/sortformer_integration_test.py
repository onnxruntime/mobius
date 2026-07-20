# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Integration test: NVIDIA NeMo Sortformer speaker diarization.

Verifies that the exported Sortformer ONNX graph produces the same per-frame
speaker-activity probabilities as the NeMo PyTorch reference model (offline
``frontend_encoder`` + ``forward_infer`` path).

Run with::

    pytest tests/sortformer_integration_test.py -m integration -sv

Skipped automatically when ``nemo_toolkit`` is not installed or the checkpoint
cannot be downloaded.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

_MODEL_ID = "nvidia/diar_streaming_sortformer_4spk-v2.1"
_NEMO_FILENAME = "diar_streaming_sortformer_4spk-v2.1.nemo"

pytest.importorskip("nemo.collections.asr", reason="nemo_toolkit not installed")
torch = pytest.importorskip("torch")


def _download_nemo() -> str:
    """Download the ``.nemo`` checkpoint, skipping the test if unavailable."""
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=_MODEL_ID, filename=_NEMO_FILENAME)
    except Exception as exc:  # pragma: no cover - network/availability guard
        pytest.skip(f"Could not download {_MODEL_ID}: {exc}")


@pytest.mark.integration
def test_sortformer_offline_parity():
    """ONNX diarization output matches the NeMo offline reference.

    The model flows through the generic :func:`build_from_nemo` pipeline:
    the ``.nemo`` archive's ``target`` (``SortformerEncLabelModel``) resolves
    to model_type ``"sortformer"`` and the ``"diarization"`` task.
    """
    from nemo.collections.asr.models import SortformerEncLabelModel

    from mobius.integrations.nemo import build_from_nemo

    nemo_path = _download_nemo()

    # --- NeMo reference (offline forward path) --------------------------------
    model = SortformerEncLabelModel.restore_from(nemo_path, map_location="cpu")
    model.eval()
    model.streaming_mode = False

    torch.manual_seed(0)
    mel = torch.randn(1, model.cfg.encoder.feat_in, 400)
    mel_len = torch.tensor([400], dtype=torch.long)

    with torch.no_grad():
        emb_seq, emb_len = model.frontend_encoder(
            processed_signal=mel, processed_signal_length=mel_len
        )
        preds_ref = model.forward_infer(emb_seq, emb_len).cpu().numpy()

    # --- mobius ONNX export ---------------------------------------------------
    pkg = build_from_nemo(nemo_path)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.onnx")
        ir.save(pkg["model"], path, external_data="model.onnx.data")
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        out = sess.run(None, {sess.get_inputs()[0].name: mel.numpy().astype(np.float32)})[0]

    assert out.shape == preds_ref.shape
    np.testing.assert_allclose(out, preds_ref, atol=1e-4)
