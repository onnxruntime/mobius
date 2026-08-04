# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""L4/L5 golden parity tests for the NeMo Sortformer speaker-diarization model.

Builds the ONNX diarization graph from the real ``.nemo`` archive
(``nvidia/diar_streaming_sortformer_4spk-v2.1``, resolved from HuggingFace Hub)
via the generic :func:`build_from_nemo` pipeline and compares its output against
a pre-computed NeMo reference
(``testdata/golden/speech/sortformer_diarization.npz``).

The reference was generated with ``nemo_toolkit`` — regenerate it with
``scripts/generate_sortformer_golden.py`` if the architecture changes.

- **L4** (``test_sortformer_diarization_parity``): the ONNX ``speaker_probs``
  match the NeMo offline reference speaker sigmoids element-wise.
- **L5** (``test_sortformer_end_to_end_diarization``): the full offline
  diarization decision (per-frame active-speaker assignment derived from the
  sigmoids) reproduces the NeMo reference over the whole utterance.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

pytestmark = pytest.mark.integration

_MODEL_ID = "nvidia/diar_streaming_sortformer_4spk-v2.1"
# Pin the HF revision so the test always validates against the exact model the
# committed golden was generated from (see scripts/generate_sortformer_golden.py).
_REVISION = "fafaab5faa1617a0ca52d38dd3dc4bd636800d3d"
_GOLDEN = os.path.join(
    os.path.dirname(__file__),
    "..",
    "testdata",
    "golden",
    "speech",
    "sortformer_diarization.npz",
)


def _run_diarizer(golden: np.lib.npyio.NpzFile) -> np.ndarray:
    """Build the ONNX diarizer via build_from_nemo and run the golden mel input."""
    from mobius.integrations.nemo import build_from_nemo

    pkg = build_from_nemo(_MODEL_ID, revision=_REVISION)
    assert "model" in pkg

    mel = golden["mel"].astype(np.float32)  # (1, feat_in, T)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "model.onnx")
        ir.save(pkg["model"], path, external_data="model.onnx.data")
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        out = sess.run(None, {sess.get_inputs()[0].name: mel})[0]
    return out


@pytest.mark.integration_slow
def test_sortformer_diarization_parity():
    """L4: ONNX speaker probabilities match the NeMo offline reference."""
    golden = np.load(_GOLDEN)
    preds_ref = golden["preds"]  # (1, T', num_spks) sigmoid activations

    out = _run_diarizer(golden)

    assert out.shape == preds_ref.shape
    # Offline fp32 forward path: tight parity (matches build_from_nemo golden).
    np.testing.assert_allclose(out, preds_ref, atol=1e-4)


@pytest.mark.integration_slow
def test_sortformer_end_to_end_diarization():
    """L5: the end-to-end offline diarization decision matches NeMo.

    The diarization output is a per-frame speaker-activity sigmoid in ``[0, 1]``.
    The task-level decisions a downstream consumer reads are (a) the dominant
    speaker per frame (``argmax``) and (b) the binarized set of active speakers
    per frame (sigmoid threshold at 0.5). Verify the ONNX pipeline reproduces
    both over the whole utterance and that the raw probabilities are well-formed.
    """
    golden = np.load(_GOLDEN)
    preds_ref = golden["preds"]
    meta = json.loads(str(golden["meta"]))
    num_spks = int(meta["num_spks"])

    out = _run_diarizer(golden)

    # Raw sigmoids must be valid probabilities of the expected speaker count.
    assert out.shape[-1] == num_spks
    assert out.min() >= 0.0 and out.max() <= 1.0

    # Diarization decision 1: per-frame dominant-speaker assignment sequence.
    np.testing.assert_array_equal(out.argmax(axis=-1), preds_ref.argmax(axis=-1))

    # Diarization decision 2: binarized per-frame active-speaker set.
    np.testing.assert_array_equal(out > 0.5, preds_ref > 0.5)
