# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for BERT/ESM/RoBERTa weight-name rename helpers.

These guard the ``preprocess_weights`` renaming used by ``BertModel`` and
``BertForMaskedLM`` so that silent mis-renames (which surface only as
mismatched initializer names during weight application) are caught early.
"""

from __future__ import annotations

import pytest

from mobius.models.bert import _rename_bert_weight, _rename_masked_lm_weight


class TestRenameBertWeight:
    @pytest.mark.parametrize(
        "hf_name,expected",
        [
            # Prefix stripping for the three supported encoder families.
            ("bert.embeddings.word_embeddings.weight", "embeddings.word_embeddings.weight"),
            ("roberta.embeddings.word_embeddings.weight", "embeddings.word_embeddings.weight"),
            ("esm.embeddings.word_embeddings.weight", "embeddings.word_embeddings.weight"),
            # Nested HF naming collapse (attention.self / attention.output).
            (
                "encoder.layer.0.attention.self.query.weight",
                "encoder.layer.0.attention.query.weight",
            ),
            (
                "encoder.layer.0.attention.output.dense.weight",
                "encoder.layer.0.attention.dense.weight",
            ),
            (
                "encoder.layer.0.output.dense.weight",
                "encoder.layer.0.dense.weight",
            ),
            # Old-BERT gamma/beta compat.
            ("encoder.layer.0.output.LayerNorm.gamma", "encoder.layer.0.LayerNorm.weight"),
            ("encoder.layer.0.output.LayerNorm.beta", "encoder.layer.0.LayerNorm.bias"),
        ],
    )
    def test_encoder_renames(self, hf_name: str, expected: str) -> None:
        assert _rename_bert_weight(hf_name) == expected

    @pytest.mark.parametrize(
        "hf_name",
        [
            "bert.pooler.dense.weight",
            "cls.predictions.decoder.weight",
            "cls.seq_relationship.weight",
        ],
    )
    def test_skipped_weights_return_none(self, hf_name: str) -> None:
        assert _rename_bert_weight(hf_name) is None


class TestRenameMaskedLMWeight:
    @pytest.mark.parametrize(
        "hf_name,expected",
        [
            # BERT-style cls.predictions.* -> lm_head.*
            ("cls.predictions.transform.dense.weight", "lm_head.dense.weight"),
            ("cls.predictions.transform.LayerNorm.weight", "lm_head.layer_norm.weight"),
            ("cls.predictions.decoder.weight", "lm_head.decoder.weight"),
            ("cls.predictions.bias", "lm_head.decoder.bias"),
            # ESM/RoBERTa lm_head.* passes through unchanged.
            ("lm_head.dense.weight", "lm_head.dense.weight"),
            ("lm_head.decoder.bias", "lm_head.decoder.bias"),
            # Encoder weights still delegate to the shared rename.
            (
                "esm.encoder.layer.0.attention.self.query.weight",
                "encoder.layer.0.attention.query.weight",
            ),
        ],
    )
    def test_masked_lm_renames(self, hf_name: str, expected: str) -> None:
        assert _rename_masked_lm_weight(hf_name) == expected

    @pytest.mark.parametrize(
        "hf_name",
        [
            "esm.pooler.dense.weight",
            "esm.contact_head.regression.weight",
            "cls.seq_relationship.weight",
        ],
    )
    def test_skipped_weights_return_none(self, hf_name: str) -> None:
        assert _rename_masked_lm_weight(hf_name) is None
