"""Numerical-parity test: HF LFM2-Audio audio encoder vs exported ONNX."""
from __future__ import annotations

import sys
import numpy as np
import soundfile as sf
import torch
import onnxruntime as ort

from liquid_audio import LFM2AudioModel
from liquid_audio.processor import LFM2AudioProcessor

REPO = "LiquidAI/LFM2-Audio-1.5B"
AUDIO_PATH = "testdata/652-129742-0006.flac"
ONNX_PATH = "/tmp/lfm2-out-fixed/model.onnx"
TARGET_SECONDS = 1.0


def main() -> int:
    torch.manual_seed(0)

    # Load HF reference (fp32, CPU for stability)
    print("Loading HF LFM2AudioModel ...", flush=True)
    model = LFM2AudioModel.from_pretrained(REPO, dtype=torch.float32, device="cpu")
    model.eval()
    proc = LFM2AudioProcessor.from_pretrained(REPO, device="cpu")
    proc.audio_processor.to(dtype=torch.float32)

    # Load audio (force 16 kHz mono via soundfile)
    audio, sr = sf.read(AUDIO_PATH, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == 16000, f"Expected 16k, got {sr}"
    n = int(sr * TARGET_SECONDS)
    audio = audio[:n]
    wav = torch.from_numpy(audio).unsqueeze(0).to(torch.float32)
    lengths = torch.tensor([wav.shape[1]], dtype=torch.long)
    print(f"Audio: shape={tuple(wav.shape)} sr={sr}")

    # Mel features via official processor
    with torch.no_grad():
        mel, mel_lens = proc.audio_processor(wav, lengths)
    print(f"Mel: shape={tuple(mel.shape)} dtype={mel.dtype} lens={mel_lens.tolist()}")

    # HF reference forward: conformer + adapter
    with torch.no_grad():
        audio_enc, audio_enc_len = model.conformer(mel, mel_lens)
        # (B, D, T) -> (B, T, D)
        audio_enc_t = audio_enc.mT
        audio_with_adapter = model.audio_adapter(audio_enc_t)
    print(
        f"HF conformer out: shape={tuple(audio_enc_t.shape)}  enc_len={audio_enc_len.tolist()}"
    )
    print(f"HF after adapter: shape={tuple(audio_with_adapter.shape)}")

    # ONNX forward
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input_features": mel.numpy()})[0]
    print(f"ONNX output: shape={ort_out.shape}")

    hf_np = audio_with_adapter.detach().numpy()

    def diff_report(a: np.ndarray, b: np.ndarray, label: str) -> None:
        print(f"\n=== {label} ===")
        print(f"  shape A={a.shape}  shape B={b.shape}")
        if a.shape != b.shape:
            print("  SHAPE MISMATCH - cannot compare element-wise")
            return
        d = np.abs(a - b)
        print(f"  max_abs_diff   = {d.max():.6e}")
        print(f"  mean_abs_diff  = {d.mean():.6e}")
        print(f"  median_abs_diff= {np.median(d):.6e}")
        print(f"  A[0,0,:5] = {a[0,0,:5]}")
        print(f"  B[0,0,:5] = {b[0,0,:5]}")
        try:
            np.testing.assert_allclose(a, b, atol=1e-3, rtol=1e-3)
            print(f"  PASS @ atol=1e-3 rtol=1e-3")
        except AssertionError as e:
            msg = str(e).splitlines()
            print(f"  FAIL @ atol=1e-3 rtol=1e-3 :: {msg[1] if len(msg)>1 else msg[0]}")

    diff_report(hf_np, ort_out, "HF (conformer+adapter) vs ONNX")

    # Isolation: compare pre-adapter
    print("\nAttempting pre-adapter isolation (ONNX = full pipeline, HF = encoder-only)")
    hf_pre = audio_enc_t.detach().numpy()
    diff_report(hf_pre, ort_out, "HF conformer-only vs ONNX  (shape mismatch likely)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
