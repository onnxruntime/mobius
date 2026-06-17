# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Map a NeMo ``model_config.yaml`` to a mobius :class:`ArchitectureConfig`.

Only the FastConformer-RNNT (``EncDecRNNTBPEModel``) family is supported for
now.  The relevant NeMo config sub-trees are::

    model_defaults: {enc_hidden, pred_hidden, joint_hidden}
    encoder:  {feat_in, n_layers, d_model, n_heads, subsampling_factor,
               subsampling_conv_channels, ff_expansion_factor,
               conv_kernel_size, conv_norm_type, pos_emb_max_len, xscaling}
    decoder:  {prednet: {pred_hidden, pred_rnn_layers}, vocab_size}
    joint:    {jointnet: {joint_hidden, encoder_hidden, pred_hidden},
               num_classes}
"""

from __future__ import annotations

import logging
from typing import Any

from mobius._configs import ArchitectureConfig

logger = logging.getLogger(__name__)

# NeMo ``target`` class path → mobius registry model_type.
NEMO_TARGET_TO_MODEL_TYPE: dict[str, str] = {
    "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel": (
        "fastconformer_rnnt"
    ),
    "nemo.collections.asr.models.rnnt_models.EncDecRNNTModel": "fastconformer_rnnt",
}


def nemo_model_type(target: str) -> str:
    """Resolve a NeMo ``target`` class path to a mobius registry model_type."""
    if target not in NEMO_TARGET_TO_MODEL_TYPE:
        raise KeyError(
            f"Unsupported NeMo target {target!r}. Supported targets: "
            f"{sorted(NEMO_TARGET_TO_MODEL_TYPE)}"
        )
    return NEMO_TARGET_TO_MODEL_TYPE[target]


def nemo_to_config(nemo_config: dict[str, Any]) -> ArchitectureConfig:
    """Build an :class:`ArchitectureConfig` from a NeMo ``model_config.yaml`` dict."""
    target = str(nemo_config.get("target", ""))
    model_type = nemo_model_type(target)

    enc = nemo_config["encoder"]
    dec = nemo_config["decoder"]
    joint = nemo_config["joint"]
    prednet = dec["prednet"]
    jointnet = joint["jointnet"]

    d_model = int(enc["d_model"])
    ff_expansion = int(enc.get("ff_expansion_factor", 4))
    # num_classes excludes the blank; the embedding/joint span num_classes + 1.
    num_classes = int(joint["num_classes"])
    vocab_with_blank = num_classes + 1

    config = ArchitectureConfig(
        vocab_size=vocab_with_blank,
        hidden_size=d_model,
        intermediate_size=d_model * ff_expansion,
        num_hidden_layers=int(enc["n_layers"]),
        num_attention_heads=int(enc["n_heads"]),
        num_key_value_heads=int(enc["n_heads"]),
        head_dim=d_model // int(enc["n_heads"]),
        # Mel-feature input dimension for the encoder subsampling stem.
        audio_input_size=int(enc["feat_in"]),
        # FastConformer encoder knobs.
        fastconformer_subsampling_factor=int(enc.get("subsampling_factor", 8)),
        fastconformer_subsampling_conv_channels=int(
            enc.get("subsampling_conv_channels", 256)
        ),
        fastconformer_conv_kernel_size=int(enc.get("conv_kernel_size", 9)),
        fastconformer_pos_emb_max_len=int(enc.get("pos_emb_max_len", 5000)),
        fastconformer_xscaling=bool(enc.get("xscaling", False)),
        # RNN-T prediction network + joint.
        rnnt_pred_hidden=int(prednet["pred_hidden"]),
        rnnt_pred_rnn_layers=int(prednet.get("pred_rnn_layers", 1)),
        rnnt_joint_hidden=int(jointnet["joint_hidden"]),
        rnnt_num_classes=num_classes,
    )
    # Stash the resolved model_type for the builder/registry lookup.
    config._nemo_model_type = model_type
    return config
