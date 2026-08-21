# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for the encoder-embedding workflow metadata producer.

A bidirectional encoder is not generative. These tests pin the three facts that
separate its metadata from every decoder's: the workflow runs once, it declares
exactly the ports the artifact exposes, and it never mentions a generation
loop, a sampler or a KV cache.
"""

from __future__ import annotations

from typing import Any

import pytest

from mobius._passes import RemoveDeadGraphInputsPass
from mobius.integrations.onnx_genai.auto_export import (
    _looks_like_encoder_embedding,
    write_onnx_genai_config,
)
from mobius.integrations.onnx_genai.workflow_metadata import (
    build_encoder_embedding_workflow_metadata,
)
from mobius.models.bert import BertModel
from mobius.models.bert_test import PROTBERT_TINY_CONFIG
from mobius.models.esm import EsmModel
from mobius.models.esm_test import TINY_CONFIG as ESM2_TINY_CONFIG
from mobius.tasks import FeatureExtractionTask


def _esm2_package():
    package = FeatureExtractionTask().build(EsmModel(ESM2_TINY_CONFIG), ESM2_TINY_CONFIG)
    # ESM-2 never reads ``token_type_ids``; the export path drops the dead input.
    RemoveDeadGraphInputsPass()(package["model"])
    return package


def _protbert_package():
    return FeatureExtractionTask().build(BertModel(PROTBERT_TINY_CONFIG), PROTBERT_TINY_CONFIG)


_PACKAGES = {"esm2": _esm2_package, "protbert": _protbert_package}


@pytest.fixture(scope="module")
def built() -> dict[str, Any]:
    return {name: build() for name, build in _PACKAGES.items()}


@pytest.fixture(params=sorted(_PACKAGES), scope="module")
def package(request, built):
    return built[request.param]


@pytest.fixture(scope="module")
def metadata(package) -> dict[str, Any]:
    return build_encoder_embedding_workflow_metadata(package, package.config)


class TestDetection:
    def test_an_embedding_encoder_is_recognized(self, package) -> None:
        assert _looks_like_encoder_embedding(package)

    def test_a_generative_package_is_not_mistaken_for_one(self) -> None:
        """A package with ``logits`` is generative however else it is shaped.

        ``logits`` is the whole signal: it is what a sampler consumes, and an
        embedding encoder has none. Renaming the port on an otherwise identical
        graph is what isolates that one fact.
        """
        package = _esm2_package()
        assert _looks_like_encoder_embedding(package)
        package["model"].graph.outputs[0].name = "logits"
        assert not _looks_like_encoder_embedding(package)

    def test_a_cached_decoder_is_not_mistaken_for_one(self) -> None:
        package = _esm2_package()
        package["model"].graph.inputs[0].name = "past_key_values.0.key"
        assert not _looks_like_encoder_embedding(package)


class TestEncoderEmbeddingWorkflow:
    def test_it_runs_exactly_once(self, metadata) -> None:
        steps = metadata["pipeline"]["workflow"]["steps"]
        assert [step["kind"] for step in steps] == ["invoke", "emit"]

    def test_it_carries_no_state(self, metadata) -> None:
        """No loop means no carried cell; a state table here would be inert."""
        assert "state" not in metadata["pipeline"]["workflow"]

    def test_it_describes_no_generation(self, metadata) -> None:
        text = repr(metadata)
        for forbidden in (
            "max_output_tokens",
            "eos_ids",
            "sampler",
            "past_key_values",
            "state_service",
            "logits",
        ):
            assert forbidden not in text, f"encoder metadata should not mention {forbidden}"

    def test_the_profile_is_an_embedding_profile(self, metadata) -> None:
        profile = metadata["profiles"]["embedding"]
        assert profile["kind"] == "embedding"
        assert profile["outputs"] == {"last_hidden_state": "last_hidden_state"}

    def test_mask_aware_pooling_is_declared_against_a_real_input(
        self, metadata, package
    ) -> None:
        profile = metadata["profiles"]["embedding"]
        workflow = metadata["pipeline"]["workflow"]
        assert profile["pooling"]["mask"] in workflow["inputs"]
        assert profile["batch_invariance"] == "row_independent"

    def test_every_bound_port_exists_in_the_artifact(self, metadata, package) -> None:
        graph = package["model"].graph
        inputs = {str(value.name) for value in graph.inputs}
        outputs = {str(value.name) for value in graph.outputs}
        invoke = metadata["pipeline"]["workflow"]["steps"][0]
        assert set(invoke["inputs"]) == inputs
        assert set(invoke["outputs"]) <= outputs

    def test_every_declared_input_is_bound(self, metadata) -> None:
        workflow = metadata["pipeline"]["workflow"]
        invoke = workflow["steps"][0]
        assert set(workflow["inputs"]) == set(invoke["inputs"].values())

    def test_the_emitted_value_is_the_invocations_output(self, metadata) -> None:
        invoke, emit = metadata["pipeline"]["workflow"]["steps"]
        assert emit["value"] in invoke["outputs"].values()
        assert emit["output"] in metadata["pipeline"]["workflow"]["outputs"]


class TestPortsFollowTheArtifact:
    """The two models disagree about ``token_type_ids``, and the metadata must.

    ESM-2 has no token-type embedding, so the feature-extraction task's third
    input is dead and gets pruned; ProtBert reads it. A producer that copied
    the task signature instead of the graph would declare a port ESM-2 does not
    expose, and a runtime would fail to bind it.
    """

    def test_esm2_declares_no_token_type_ids(self, built) -> None:
        metadata = build_encoder_embedding_workflow_metadata(built["esm2"])
        assert "request.token_type_ids" not in metadata["pipeline"]["workflow"]["inputs"]

    def test_protbert_declares_token_type_ids(self, built) -> None:
        metadata = build_encoder_embedding_workflow_metadata(built["protbert"])
        assert "request.token_type_ids" in metadata["pipeline"]["workflow"]["inputs"]


class TestDispatch:
    def test_the_export_entry_point_picks_the_encoder_builder(self, tmp_path) -> None:
        """Without this, an encoder falls through to the decoder fallback."""
        package = _esm2_package()
        artifacts = write_onnx_genai_config(package, str(tmp_path), config=package.config)
        with open(artifacts["inference_metadata"], encoding="utf-8") as handle:
            text = handle.read()
        assert "kind: embedding" in text
        assert "max_output_tokens" not in text


class TestRejections:
    def test_a_package_without_last_hidden_state_is_refused(self) -> None:
        package = _esm2_package()
        package["model"].graph.outputs[0].name = "not_hidden_states"
        with pytest.raises(ValueError, match="last_hidden_state"):
            build_encoder_embedding_workflow_metadata(package)
