# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shared audio preprocessing utilities for ASR examples.

Provides the LFR fbank frontend pipeline shared by Fun-ASR-Nano and
SenseVoiceSmall: audio loading, mel spectrogram, LFR stacking, and CMVN.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
LFR_M = 7  # LFR stack factor
LFR_N = 6  # LFR stride
N_MELS = 80


def load_audio_file(path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio file and resample to target sample rate (mono, float32)."""
    import torchaudio

    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform.squeeze(0).numpy().astype(np.float32)


def compute_fbank(
    audio: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
) -> np.ndarray:
    """Compute log-mel filterbank features using torchaudio (Kaldi-compatible).

    Args:
        audio: 1-D waveform array (float32, mono).
        sample_rate: Audio sample rate in Hz.
        n_mels: Number of mel filter bank bins.
        frame_length: Frame length in ms.
        frame_shift: Frame shift (hop) in ms.

    Returns:
        ``(T, n_mels)`` fbank feature matrix.
    """
    import torch
    import torchaudio

    wav = torch.from_numpy(audio).float().unsqueeze(0)
    fbank = torchaudio.compliance.kaldi.fbank(
        wav,
        num_mel_bins=n_mels,
        sample_frequency=sample_rate,
        window_type="hamming",
        frame_length=frame_length,
        frame_shift=frame_shift,
        dither=0.0,
    )
    return fbank.numpy()


def apply_lfr(fbank: np.ndarray, lfr_m: int = LFR_M, lfr_n: int = LFR_N) -> np.ndarray:
    """Apply Low Frame Rate stacking and subsampling.

    Stacks ``lfr_m`` consecutive frames and subsamples every ``lfr_n`` frames,
    producing features of dimension ``lfr_m * n_mels`` (typically 7*80 = 560).

    Left-pads by ``(lfr_m - 1) // 2`` frames (FunASR convention) so that the
    first output frame is centered on the first input frame.

    Returns array of shape ``(T_out, lfr_m * n_mels)``.
    """
    # Left-pad by (lfr_m - 1) // 2 frames (FunASR convention)
    left_pad = (lfr_m - 1) // 2  # = 3 for lfr_m=7
    fbank = np.pad(fbank, ((left_pad, 0), (0, 0)), mode="edge")

    num_frames = fbank.shape[0]
    pad_len = (lfr_n - (num_frames % lfr_n)) % lfr_n
    if pad_len > 0:
        fbank = np.pad(fbank, ((0, pad_len), (0, 0)), mode="edge")
    t_padded = fbank.shape[0]

    lfr_frames = []
    for i in range(0, t_padded, lfr_n):
        end = min(i + lfr_m, t_padded)
        chunk = fbank[i:end]
        if chunk.shape[0] < lfr_m:
            chunk = np.pad(chunk, ((0, lfr_m - chunk.shape[0]), (0, 0)), mode="edge")
        lfr_frames.append(chunk.flatten())
    return np.array(lfr_frames)


def load_cmvn(cmvn_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load CMVN stats from Kaldi am.mvn file.

    Returns (means, vars) arrays of shape ``(560,)`` each.
    CMVN is applied as: ``features = (features + means) * vars``
    """
    with open(cmvn_path) as f:
        lines = f.readlines()
    means = None
    variances = None
    for i, line in enumerate(lines):
        parts = line.split()
        if parts[0] == "<AddShift>":
            next_parts = lines[i + 1].split()
            if next_parts[0] == "<LearnRateCoef>":
                means = np.array(next_parts[3:-1], dtype=np.float32)
        elif parts[0] == "<Rescale>":
            next_parts = lines[i + 1].split()
            if next_parts[0] == "<LearnRateCoef>":
                variances = np.array(next_parts[3:-1], dtype=np.float32)
    if means is None or variances is None:
        raise ValueError(f"Failed to parse CMVN from {cmvn_path}")
    return means, variances


def preprocess_audio(
    audio: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    cmvn: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Full frontend: fbank → LFR → CMVN → ``(1, T, 560)``.

    Args:
        audio: 1-D waveform (float32, mono, at ``sample_rate``).
        sample_rate: Audio sample rate.
        n_mels: Mel bins.
        cmvn: Optional ``(means, vars)`` from :func:`load_cmvn`.

    Returns:
        ``(1, T_lfr, lfr_m * n_mels)`` feature tensor.
    """
    fbank = compute_fbank(audio, sample_rate=sample_rate, n_mels=n_mels)
    lfr = apply_lfr(fbank)
    if cmvn is not None:
        means, variances = cmvn
        lfr = (lfr + means) * variances
    return lfr[np.newaxis, :, :]
