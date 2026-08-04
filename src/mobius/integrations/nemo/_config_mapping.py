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

from typing import Any

from mobius._configs import ArchitectureConfig
from mobius._configs._base import BaseModelConfig

# NeMo ``target`` class path → mobius registry model_type.
NEMO_TARGET_TO_MODEL_TYPE: dict[str, str] = {
    "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel": "fastconformer_rnnt",
    "nemo.collections.asr.models.rnnt_models.EncDecRNNTModel": "fastconformer_rnnt",
    "nemo.collections.asr.models.sortformer_diar_models.SortformerEncLabelModel": "sortformer",
}


def nemo_model_type(target: str) -> str:
    """Resolve a NeMo ``target`` class path to a mobius registry model_type."""
    if target not in NEMO_TARGET_TO_MODEL_TYPE:
        raise KeyError(
            f"Unsupported NeMo target {target!r}. Supported targets: "
            f"{sorted(NEMO_TARGET_TO_MODEL_TYPE)}"
        )
    return NEMO_TARGET_TO_MODEL_TYPE[target]


def _validate_encoder(enc: dict[str, Any]) -> None:
    """Reject NeMo encoder configs whose architecture this builder cannot emit.

    The FastConformer builder hard-codes several architectural choices
    (8x dw_striding causal subsampling, relative-position attention,
    layer-norm conv module, no input scaling).  A ``.nemo`` file that uses
    different settings would load weights yet produce silently wrong outputs,
    so fail loudly instead.
    """
    checks = {
        "subsampling": ("dw_striding", str(enc.get("subsampling", "dw_striding"))),
        "subsampling_factor": (8, int(enc.get("subsampling_factor", 8))),
        "self_attention_model": (
            "rel_pos",
            str(enc.get("self_attention_model", "rel_pos")),
        ),
        "conv_norm_type": (
            "layer_norm",
            str(enc.get("conv_norm_type", "layer_norm")),
        ),
        "xscaling": (False, bool(enc.get("xscaling"))),
        # The conformer stack hard-codes bias=False (FF / q,k,v,out / conv).
        # NeMo defaults use_bias=True; a default config would load weights but
        # silently drop every bias, producing wrong output. Reject it loudly.
        "use_bias": (False, bool(enc.get("use_bias", True))),
        # The masks implement NeMo's chunked_limited rule only; other styles
        # ("regular", "chunked_limited_with_rc") would build a wrong mask.
        "att_context_style": (
            "chunked_limited",
            str(enc.get("att_context_style", "chunked_limited")),
        ),
    }
    unsupported = {
        key: actual for key, (expected, actual) in checks.items() if actual != expected
    }
    if unsupported:
        details = ", ".join(
            f"{key}={actual!r} (supported: {checks[key][0]!r})"
            for key, actual in unsupported.items()
        )
        raise ValueError(
            "Unsupported FastConformer encoder configuration: "
            f"{details}. Only the offline dw_striding/rel_pos/layer_norm "
            "FastConformer-RNNT variant is currently supported."
        )


def nemo_to_config(nemo_config: dict[str, Any]) -> BaseModelConfig:
    """Build a mobius config from a NeMo ``model_config.yaml`` dict.

    Dispatches on the NeMo ``target`` class path: FastConformer-RNNT models
    produce an :class:`ArchitectureConfig`; Sortformer diarization models
    produce a :class:`SortformerConfig`.
    """
    target = str(nemo_config.get("target", ""))
    model_type = nemo_model_type(target)

    if model_type == "sortformer":
        # Imported lazily to avoid a models→integrations import cycle.
        from mobius.models.sortformer import SortformerConfig

        return SortformerConfig.from_nemo_yaml(nemo_config)

    enc = nemo_config["encoder"]
    dec = nemo_config["decoder"]
    joint = nemo_config["joint"]
    prednet = dec["prednet"]
    jointnet = joint["jointnet"]

    _validate_encoder(enc)

    d_model = int(enc["d_model"])
    ff_expansion = int(enc.get("ff_expansion_factor", 4))
    # num_classes excludes the blank; the embedding/joint span num_classes + 1.
    num_classes = int(joint["num_classes"])
    vocab_with_blank = num_classes + 1
    # Chunked-limited streaming attention context [left, right] in frames.
    # NeMo may store a list of contexts (cache-aware multi-context training);
    # use the first / primary one.
    att_context_size = enc.get("att_context_size", [70, 13])
    if att_context_size and isinstance(att_context_size[0], (list, tuple)):
        att_context_size = att_context_size[0]
    att_context = (int(att_context_size[0]), int(att_context_size[1]))
    # A finite right context is required: chunk_size = right + 1, and the
    # "unlimited right" sentinel (-1) would make chunk_size 0 → div-by-zero in
    # the streaming mask. (left == -1 is handled: it clamps to cache_size 70.)
    if att_context[1] < 0:
        raise ValueError(
            "Unsupported FastConformer att_context_size right context "
            f"{att_context[1]} (must be >= 0); 'unlimited right' streaming is "
            "not supported."
        )

    # Cache-aware streaming sizes (NeMo ``setup_streaming_params``):
    # last_channel_cache_size = att_context left; drop_extra_pre_encoded is a
    # function of the subsampling pre-encode cache (= 9 for 8x dw_striding).
    subsampling_factor = int(enc.get("subsampling_factor", 8))
    streaming_cache_size = att_context[0] if att_context[0] >= 0 else 70
    # NeMo may store pre_encode_cache_size as a list (cache-aware multi-context
    # training); use the last (largest) entry, mirroring the GenAI config path.
    pre_encode_cache = enc.get("pre_encode_cache_size", subsampling_factor + 1)
    if isinstance(pre_encode_cache, (list, tuple)):
        pre_encode_cache = pre_encode_cache[-1]
    pre_encode_cache = int(pre_encode_cache)
    drop_extra = (
        1 + (pre_encode_cache - 1) // subsampling_factor if pre_encode_cache >= 1 else 0
    )

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
        fastconformer_feat_in=int(enc["feat_in"]),
        fastconformer_att_context_size=att_context,
        fastconformer_streaming_cache_size=streaming_cache_size,
        fastconformer_streaming_drop_extra=drop_extra,
        # FastConformer encoder knobs.
        fastconformer_subsampling_factor=int(enc.get("subsampling_factor", 8)),
        fastconformer_subsampling_conv_channels=int(enc.get("subsampling_conv_channels", 256)),
        fastconformer_conv_kernel_size=int(enc.get("conv_kernel_size", 9)),
        fastconformer_pos_emb_max_len=int(enc.get("pos_emb_max_len", 5000)),
        fastconformer_xscaling=bool(enc.get("xscaling", False)),
        # RNN-T prediction network + joint.
        rnnt_pred_hidden=int(prednet["pred_hidden"]),
        rnnt_pred_rnn_layers=int(prednet.get("pred_rnn_layers", 1)),
        rnnt_joint_hidden=int(jointnet["joint_hidden"]),
        rnnt_num_classes=num_classes,
        # Resolved registry model_type, stored on the native field so it
        # survives dataclasses.replace() (e.g. the builder's dtype swap).
        model_type=model_type,
    )
    return config
