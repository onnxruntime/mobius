# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pinned upstream Sigma reference loading, independent of the Mobius graph."""

from __future__ import annotations

import inspect

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "nvidia/Nemotron-Labs-Sigma-distill-spec-3B"
REVISION = "9769b2d388000e0ff61736d26359074efbf0dc9d"


def load_reference():
    """Reconstruct only the upstream nonpersistent buffer lost by HF meta load."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Sigma reference qualification requires CUDA; select an idle GPU "
            "using CUDA_VISIBLE_DEVICES before starting Python."
        )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = (
        AutoModel.from_pretrained(
            MODEL_ID, revision=REVISION, trust_remote_code=True, dtype=torch.float32
        )
        .cuda()
        .eval()
    )
    upstream = inspect.getmodule(model)
    # HF meta initialization materializes nonpersistent buffers without values;
    # the source _init_weights is a no-op. All parameters and math stay unchanged.
    model.encoder.rotary_emb = upstream.NLCDRotaryEmbedding(model.config).cuda()
    if not model.encoder.rotary_emb.inv_freq.isfinite().all():
        raise RuntimeError(
            f"Upstream Sigma RoPE initialization is nonfinite at {REVISION}; "
            "check the pinned Transformers environment before generating goldens."
        )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True
    )
    return model, tokenizer
