# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for the ESM-2 protein encoder.

These build the ONNX graph from a tiny config -- no weights, no network -- and
assert the structural facts that separate ESM-2 from a BERT clone: rotary
positions instead of a learned position table, no token-type embedding, and a
final ``emb_layer_norm_after``.
"""

from __future__ import annotations

import onnx_ir as ir
import pytest

from mobius.models.esm import EsmConfig, EsmModel, _rename_esm_weight
from mobius.tasks import FeatureExtractionTask

#: Shape-faithful miniature of ``facebook/esm2_t6_8M_UR50D``: same vocabulary,
#: same special-token ids and the same architectural switches, with the widths
#: shrunk so the graph builds in well under a second.
TINY_CONFIG = EsmConfig(
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    intermediate_size=128,
    vocab_size=33,
    max_position_embeddings=1026,
    hidden_act="gelu",
    rms_norm_eps=1e-5,
    position_embedding_type="rotary",
    emb_layer_norm_before=False,
    token_dropout=True,
    mask_token_id=32,
    pad_token_id=1,
    rope_type="default",
    rope_theta=10000.0,
    partial_rotary_factor=1.0,
)


@pytest.fixture(scope="module")
def package():
    return FeatureExtractionTask().build(EsmModel(TINY_CONFIG), TINY_CONFIG)


class TestEsmGraph:
    def test_emits_per_residue_embeddings(self, package) -> None:
        model = package["model"]
        outputs = {str(value.name): value for value in model.graph.outputs}
        assert set(outputs) == {"last_hidden_state"}
        shape = [str(dim) for dim in outputs["last_hidden_state"].shape]
        assert shape[-1] == str(TINY_CONFIG.hidden_size)
        assert outputs["last_hidden_state"].dtype == ir.DataType.FLOAT

    def test_declares_no_learned_position_table(self, package) -> None:
        """ESM-2 is rotary; a learned position table would be dead weight."""
        names = set(package["model"].graph.initializers)
        assert not any("position_embeddings" in name for name in names)

    def test_declares_no_token_type_embedding(self, package) -> None:
        names = set(package["model"].graph.initializers)
        assert not any("token_type_embeddings" in name for name in names)

    def test_closes_the_stack_with_a_final_layer_norm(self, package) -> None:
        names = set(package["model"].graph.initializers)
        assert "encoder.emb_layer_norm_after.weight" in names

    def test_shares_one_rotary_cache_across_layers(self, package) -> None:
        """Every ESM-2 layer uses identical ``inv_freq``, so one cache suffices."""
        names = set(package["model"].graph.initializers)
        caches = {name for name in names if "cos_cache" in name or "sin_cache" in name}
        assert caches == {"rotary_emb.cos_cache", "rotary_emb.sin_cache"}

    def test_initializer_names_match_huggingface(self, package) -> None:
        """The renamer only strips a prefix, so the scopes must already agree."""
        names = set(package["model"].graph.initializers)
        for expected in (
            "embeddings.word_embeddings.weight",
            "encoder.layer.0.attention.self.query.weight",
            "encoder.layer.0.attention.output.dense.weight",
            "encoder.layer.0.intermediate.dense.weight",
            "encoder.layer.0.output.dense.weight",
        ):
            assert expected in names


class TestRenameEsmWeight:
    @pytest.mark.parametrize(
        "hf_name,expected",
        [
            ("esm.embeddings.word_embeddings.weight", "embeddings.word_embeddings.weight"),
            (
                "esm.encoder.layer.0.attention.self.query.weight",
                "encoder.layer.0.attention.self.query.weight",
            ),
            (
                "esm.encoder.layer.0.attention.output.dense.bias",
                "encoder.layer.0.attention.output.dense.bias",
            ),
            ("esm.encoder.emb_layer_norm_after.weight", "encoder.emb_layer_norm_after.weight"),
            # A checkpoint saved without the task prefix passes through.
            ("encoder.layer.1.output.dense.weight", "encoder.layer.1.output.dense.weight"),
        ],
    )
    def test_renames(self, hf_name: str, expected: str) -> None:
        assert _rename_esm_weight(hf_name) == expected

    @pytest.mark.parametrize(
        "hf_name",
        [
            # Rotary is recomputed as a cache, absolute positions are unused,
            # and the heads belong to other tasks.
            "esm.embeddings.position_embeddings.weight",
            "esm.embeddings.position_ids",
            "esm.encoder.layer.0.attention.self.rotary_embeddings.inv_freq",
            "esm.pooler.dense.weight",
            "lm_head.decoder.weight",
            "esm.contact_head.regression.weight",
        ],
    )
    def test_skipped_weights_return_none(self, hf_name: str) -> None:
        assert _rename_esm_weight(hf_name) is None
