# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Integration test: NVIDIA NeMo streaming multi-talker Parakeet RNN-T.

Verifies that the three exported ONNX graphs (encoder / decoder / joint)
reproduce the NeMo PyTorch reference for the cache-aware streaming
``EncDecMultiTalkerRNNTBPEModel`` inference path:

- encoder: ``forward_for_export`` with streaming channel/time caches and
  speaker-kernel injection,
- decoder: ``RNNTDecoder.predict`` (LSTM prediction network),
- joint: ``RNNTJoint.joint`` (eval-mode log-softmax).

Run with::

    pytest tests/parakeet_multitalker_integration_test.py -m integration -sv

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

_MODEL_ID = "nvidia/multitalker-parakeet-streaming-0.6b-v1"
_NEMO_FILENAME = "multitalker-parakeet-streaming-0.6b-v1.nemo"

pytest.importorskip("nemo.collections.asr", reason="nemo_toolkit not installed")
torch = pytest.importorskip("torch")


def _download_nemo() -> str:
    """Download the ``.nemo`` checkpoint, skipping the test if unavailable."""
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=_MODEL_ID, filename=_NEMO_FILENAME)
    except Exception as exc:  # pragma: no cover - network/availability guard
        pytest.skip(f"Could not download {_MODEL_ID}: {exc}")


def _log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def _sess(pkg, name: str, tmp: str) -> ort.InferenceSession:
    path = os.path.join(tmp, f"{name}.onnx")
    ir.save(pkg[name], path, external_data=f"{name}.onnx.data")
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


@pytest.mark.integration
def test_parakeet_multitalker_streaming_parity():
    """ONNX encoder/decoder/joint match the NeMo streaming reference."""
    from nemo.collections.asr.models import EncDecMultiTalkerRNNTBPEModel

    from mobius.models.parakeet_multitalker import build_parakeet_multitalker

    nemo_path = _download_nemo()

    # --- NeMo reference (cache-aware streaming) -------------------------------
    m = EncDecMultiTalkerRNNTBPEModel.restore_from(nemo_path, map_location="cpu")
    m.eval()
    enc = m.encoder
    enc.set_default_att_context_size([70, 13])
    enc.setup_streaming_params()

    cc, ct, ccl = enc.get_initial_cache_state(batch_size=1)
    cc_in = cc.transpose(0, 1).contiguous()
    ct_in = ct.transpose(0, 1).contiguous()

    torch.manual_seed(0)
    t_mel = 65
    mel = torch.randn(1, 128, t_mel)
    length = torch.tensor([t_mel], dtype=torch.int64)

    t_out = 7
    torch.manual_seed(1)
    spk = (torch.rand(1, t_out) > 0.3).float()
    bg = (torch.rand(1, t_out) > 0.5).float()
    m.set_speaker_targets(spk_targets=spk, bg_spk_targets=bg)

    with torch.no_grad():
        enc_out, enc_len, cc_next, ct_next, ccl_next = enc.forward_for_export(
            audio_signal=mel,
            length=length,
            cache_last_channel=cc_in,
            cache_last_time=ct_in,
            cache_last_channel_len=ccl,
        )

    dec = m.decoder
    tokens = torch.tensor([[3, 10, 25, 7]], dtype=torch.long)
    h0 = torch.zeros(2, 1, 640)
    c0 = torch.zeros(2, 1, 640)
    with torch.no_grad():
        g, (h1, c1) = dec.predict(tokens, state=[h0, c0], add_sos=False, batch_size=1)

    joint = m.joint
    f = enc_out[:, :, :1].transpose(1, 2)
    gg = g[:, :1, :]
    with torch.no_grad():
        joint_logits = joint.joint(f, gg).cpu().numpy()

    # --- mobius ONNX export ---------------------------------------------------
    pkg = build_parakeet_multitalker(nemo_path)

    with tempfile.TemporaryDirectory() as tmp:
        enc_sess = _sess(pkg, "encoder", tmp)
        dec_sess = _sess(pkg, "decoder", tmp)
        jt_sess = _sess(pkg, "joint", tmp)

        # Encoder: NeMo native audio_signal is [B, feat, T] -> [B, T, feat].
        enc_res = dict(
            zip(
                [o.name for o in enc_sess.get_outputs()],
                enc_sess.run(
                    None,
                    {
                        "audio_signal": mel.numpy().transpose(0, 2, 1).copy(),
                        "length": length.numpy(),
                        "cache_last_channel": cc_in.numpy(),
                        "cache_last_time": ct_in.numpy(),
                        "cache_last_channel_len": ccl.numpy(),
                        "spk_mask": spk.numpy(),
                        "bg_mask": bg.numpy(),
                    },
                ),
            )
        )
        # NeMo enc_out is [B, D, T]; the ONNX graph emits [B, T, D].
        np.testing.assert_allclose(
            enc_res["outputs"],
            enc_out.numpy().transpose(0, 2, 1),
            atol=1e-4,
        )
        np.testing.assert_array_equal(enc_res["encoded_lengths"], enc_len.numpy())
        np.testing.assert_allclose(
            enc_res["cache_last_channel_next"], cc_next.numpy(), atol=1e-4
        )
        np.testing.assert_allclose(
            enc_res["cache_last_time_next"], ct_next.numpy(), atol=1e-3
        )
        np.testing.assert_array_equal(
            enc_res["cache_last_channel_len_next"], ccl_next.numpy()
        )

        # Decoder: NeMo decoder_output is [B, H, U]; internal g is [B, U, H].
        dec_res = dict(
            zip(
                [o.name for o in dec_sess.get_outputs()],
                dec_sess.run(
                    None,
                    {
                        "targets": tokens.numpy(),
                        "h_in": h0.numpy(),
                        "c_in": c0.numpy(),
                    },
                ),
            )
        )
        np.testing.assert_allclose(
            dec_res["decoder_output"], g.numpy().transpose(0, 2, 1), atol=1e-4
        )
        np.testing.assert_allclose(dec_res["h_out"], h1.numpy(), atol=1e-4)
        np.testing.assert_allclose(dec_res["c_out"], c1.numpy(), atol=1e-4)

        # Joint: NeMo applies log-softmax over the vocab in eval mode.
        jt_out = jt_sess.run(
            None,
            {
                "encoder_output": enc_out.numpy().transpose(0, 2, 1)[:, :1, :].copy(),
                "decoder_output": g.numpy()[:, :1, :].copy(),
            },
        )[0]
        np.testing.assert_allclose(_log_softmax(jt_out), joint_logits, atol=1e-4)
