# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the NeMo config → ArchitectureConfig mapping."""

from __future__ import annotations

import pytest

from mobius.integrations.nemo._config_mapping import (
    nemo_model_type,
    nemo_to_config,
)

_RNNT_CONFIG = {
    "target": "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel",
    "encoder": {
        "feat_in": 128,
        "n_layers": 24,
        "d_model": 1024,
        "n_heads": 8,
        "subsampling_factor": 8,
        "subsampling_conv_channels": 256,
        "ff_expansion_factor": 4,
        "conv_kernel_size": 9,
        "pos_emb_max_len": 5000,
        "xscaling": False,
    },
    "decoder": {"prednet": {"pred_hidden": 640, "pred_rnn_layers": 2}, "vocab_size": 1024},
    "joint": {
        "jointnet": {"joint_hidden": 640, "encoder_hidden": 1024, "pred_hidden": 640},
        "num_classes": 1024,
    },
}


class TestNemoToConfig:
    def test_maps_core_dimensions(self):
        config = nemo_to_config(_RNNT_CONFIG)
        assert config._nemo_model_type == "fastconformer_rnnt"
        # vocab includes the blank symbol.
        assert config.vocab_size == 1025
        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 24
        assert config.num_attention_heads == 8
        assert config.head_dim == 128
        # ff_expansion_factor * d_model.
        assert config.intermediate_size == 4096

    def test_maps_fastconformer_and_rnnt_fields(self):
        config = nemo_to_config(_RNNT_CONFIG)
        assert config.audio_input_size == 128
        assert config.fastconformer_subsampling_factor == 8
        assert config.fastconformer_subsampling_conv_channels == 256
        assert config.fastconformer_conv_kernel_size == 9
        assert config.fastconformer_pos_emb_max_len == 5000
        assert config.rnnt_pred_hidden == 640
        assert config.rnnt_pred_rnn_layers == 2
        assert config.rnnt_joint_hidden == 640
        assert config.rnnt_num_classes == 1024

    def test_unsupported_target_raises(self):
        with pytest.raises(KeyError, match="Unsupported NeMo target"):
            nemo_model_type("nemo.collections.asr.models.SomethingElse")

    def test_default_ff_expansion(self):
        cfg = {**_RNNT_CONFIG, "encoder": {**_RNNT_CONFIG["encoder"]}}
        del cfg["encoder"]["ff_expansion_factor"]
        config = nemo_to_config(cfg)
        assert config.intermediate_size == 1024 * 4
