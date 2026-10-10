"""Numerical reports for logits and intermediate tensors.

Top-five output is only a display limit: error metrics compare the entire
last-token logit vector. These reports are diagnostics, not pass/fail gates.
"""

from __future__ import annotations

import numpy as np


def compare_logits(label, actual, expected, tokenizer):
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if actual.shape != expected.shape:
        raise ValueError(f"{label}: logit shapes differ: {actual.shape} vs {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError(f"{label}: non-finite logits")
    difference = np.abs(actual - expected)
    correlation = (
        float(np.corrcoef(actual, expected)[0, 1])
        if actual.std() > 0 and expected.std() > 0
        else float("nan")
    )
    print(
        f"\n{label}: mean_abs={difference.mean():.6f}; "
        f"max_abs={difference.max():.6f}; correlation={correlation:.6f}"
    )
    print(f"  Top-1 match: {actual.argmax() == expected.argmax()}")
    for source, logits in (("TRT", actual), ("HF", expected)):
        top = np.argsort(logits)[-5:][::-1]
        predictions = ", ".join(
            f"{int(token)} {tokenizer.decode([int(token)])!r} ({logits[token]:.4f})"
            for token in top
        )
        print(f"  {source} top-5: {predictions}", flush=True)


def compare_tensor(label, actual, expected):
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if actual.shape != expected.shape:
        raise ValueError(f"{label}: shapes differ: {actual.shape} vs {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError(f"{label}: non-finite values")
    error = actual - expected
    relative_rms = np.linalg.norm(error) / max(float(np.linalg.norm(expected)), 1e-12)
    print(
        f"  {label}: mean_abs={np.abs(error).mean():.6f}; "
        f"max_abs={np.abs(error).max():.6f}; relative_rms={relative_rms:.6f}",
        flush=True,
    )


class HuggingFaceReference:
    """Run identical inputs on CPU while keeping an independent HF cache."""

    def __init__(self, model_id, revision=None):
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.cache = None
        print("Loading HuggingFace FP32/eager reference on CPU...", flush=True)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                dtype=torch.float32,
                attn_implementation="eager",
            )
            .cpu()
            .eval()
        )
        print(f"Reference revision: {getattr(self.model.config, '_commit_hash', None)}")
        print("Ensure this checkpoint matches the export; engine provenance is not verified.")
        print(
            "Both runtimes receive TRT-selected tokens; metrics are diagnostic, not a parity gate."
        )

    def reset_cache(self):
        self.cache = None

    def forward(self, feeds):
        torch = self.torch
        valid_length = int(feeds["nonpad_kv_seqlen"][0])
        with torch.inference_mode():
            output = self.model(
                input_ids=torch.from_numpy(feeds["input_ids"].copy()),
                position_ids=torch.from_numpy(feeds["position_ids"].copy()),
                attention_mask=torch.ones((1, valid_length), dtype=torch.long),
                past_key_values=self.cache,
                use_cache=True,
            )
        self.cache = output.past_key_values
        return output.logits[0, -1].float().cpu().numpy()
