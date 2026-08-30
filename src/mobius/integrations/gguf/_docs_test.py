# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Closure tests for the generated GGUF capability catalog."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mobius.integrations.gguf._arch_registry import iter_arch_specs
from mobius.integrations.gguf._docs import (
    _architecture_reason,
    check_document,
    render_blocks,
    render_document,
    update_document,
)
from mobius.integrations.gguf._mmproj_registry import (
    MMPROJ_ARTIFACT_AVAILABILITY_PINS,
    MMPROJ_ARTIFACT_PINS,
    iter_projector_specs,
)
from mobius.integrations.gguf._quant_capabilities import (
    check_quantization_capability_matrix,
)
from mobius.integrations.gguf._quant_registry import iter_quant_specs
from mobius.integrations.gguf._spec import StorageRole, Support
from mobius.integrations.gguf._tokenizer_registry import tokenizer_pre_policies
from mobius.integrations.gguf._upstream import UPSTREAM_COMMIT, UPSTREAM_DATE


def test_catalog_is_exact_generator_output() -> None:
    assert check_document()
    assert check_quantization_capability_matrix()


def test_catalog_is_complete_and_reason_coded() -> None:
    document = Path("docs/gguf-capability-catalog.md").read_text(encoding="utf-8")
    assert len(document.splitlines()) < 600
    assert "RUNTIME_EVIDENCE_PENDING" in document
    assert "qwen3.5-0.8b-q4-tokenizer" in document
    assert "qwen2.5-0.5b-instruct-q8-tokenizer" in document
    assert "deferred-compiled-semantics" in document
    assert "deferred-pinned-artifact-mismatch" in document
    assert "does not claim graph or runtime support." in " ".join(document.split())


def test_user_guide_is_concise_and_links_catalog() -> None:
    document = Path("docs/api/build_from_gguf.md").read_text(encoding="utf-8")
    assert len(document.splitlines()) <= 200
    assert "../gguf-capability-catalog.md" in document
    assert "mobius build-gguf GGUF_PATH --output OUTPUT_DIR [options]" in document
    assert "runtime_validation_status" in document
    assert "<!-- BEGIN GGUF" not in document


def test_tokenizer_evidence_rows_have_exactly_four_markdown_columns() -> None:
    evidence_ids = {
        "lfm2-350m-f16-tokenizer",
        "qwen2.5-0.5b-instruct-q8-tokenizer",
        "qwen3.5-0.8b-q4-tokenizer",
        "smollm-135m-f16-tokenizer",
    }
    rows = [
        line
        for line in render_document().splitlines()
        if line.startswith("| `") and line.split("`", 2)[1] in evidence_ids
    ]
    assert len(rows) == 4
    for row in rows:
        parts = re.split(r"(?<!\\)\|", row)
        assert parts[0] == parts[-1] == ""
        assert len([part.strip() for part in parts[1:-1]]) == 4


def test_architecture_restrictions_are_concise_and_fully_registry_derived() -> None:
    rows = {
        line.split("`", 2)[1]: line
        for line in render_blocks()["architectures"].splitlines()
        if line.startswith("| `")
    }
    for spec in iter_arch_specs():
        reason = _architecture_reason(spec)
        assert rows[spec.gguf_arch].endswith(f"| {reason} |")
        assert len(reason) <= 320
        restriction = reason.partition(" — ")[2]
        assert restriction
        assert len(restriction.split(". ")) == 1


def test_ernie45_row_preserves_exact_admitted_subset_boundary() -> None:
    ernie = next(spec for spec in iter_arch_specs() if spec.gguf_arch == "ernie4_5")
    assert _architecture_reason(ernie) == (
        "RUNTIME_EVIDENCE_PENDING / FLOAT_IMPORT_ONLY — Import is narrowed to the dense "
        "split-Q/K/V, split-SwiGLU, full-RoPE variant and rejects all expert, fused, "
        "sectioned-position, and bias alternatives."
    )


def test_manual_real_artifact_table_covers_every_registry_pin() -> None:
    document = Path("docs/gguf-capability-catalog.md").read_text(encoding="utf-8")
    for pin in MMPROJ_ARTIFACT_PINS:
        assert f"`{pin.repository}@{pin.revision}`" in document
        assert f"`{pin.filename}`" in document
        assert f"`{pin.lfs_sha256}`" in document
        assert f"{pin.size:,}" in document


def test_stale_at_style_pin_is_rejected(tmp_path: Path) -> None:
    document = tmp_path / "gguf-capability-catalog.md"
    document.write_text(
        Path("docs/gguf-capability-catalog.md").read_text(encoding="utf-8")
        + "\nllama.cpp@1111111111111111111111111111111111111111\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"Stale llama\.cpp pins"):
        update_document(document)


def test_candidate_projector_artifact_table_covers_every_availability_pin() -> None:
    document = Path("docs/gguf-capability-catalog.md").read_text(encoding="utf-8")
    for pin in MMPROJ_ARTIFACT_AVAILABILITY_PINS:
        assert f"`{pin.repository}@{pin.revision}`" in document
        assert f"`{pin.filename}`" in document
        assert f"`{pin.lfs_sha256}`" in document
        assert f"{pin.size:,}" in document


def test_generated_census_counts_and_pin_are_closed() -> None:
    blocks = render_blocks()
    assert UPSTREAM_COMMIT in blocks["summary"]
    assert UPSTREAM_DATE in blocks["summary"]
    assert blocks["architectures"].count("\n| `") == 148
    assert blocks["qtypes"].count("\n| ") == 25
    assert blocks["projectors"].count("\n| `") == 60
    assert blocks["tokenizers"].count("\n| `") == 87
    assert len({policy.canonical for policy in tokenizer_pre_policies().values()}) == 56


def test_runtime_support_requires_structured_evidence() -> None:
    supported = [spec for spec in iter_arch_specs() if spec.runtime is Support.SUPPORTED]
    assert [(spec.gguf_arch, spec.runtime_evidence_ids) for spec in supported] == [
        (
            "llama",
            (
                "smollm-135m-f16-onnxruntime-1.29.0",
                "smollm-135m-f16-ort-genai-0.15.2",
            ),
        ),
        ("qwen2", ("qwen2.5-0.5b-instruct-q8-ort-genai-0.15.2",)),
        ("lfm2", ("lfm2-350m-f16-ort-genai-0.15.2",)),
        ("qwen35moe", ("qwen3.5-moe-0.87b-q2-k-ort-genai-0.15.2",)),
        ("gpt2", ("gpt2-q2-k-ort-genai-0.15.2",)),
        ("starcoder2", ("tiny-starcoder2-q2-k-ort-genai-0.15.2",)),
        ("olmo", ("tiny-olmo-q2-k-ort-genai-0.15.2",)),
        (
            "apertus",
            ("apertus-v1.1-1.5b-instruct-bf16-ort-genai-0.15.2",),
        ),
        ("mpt", ("tiny-mpt-q2-k-ort-genai-0.15.2",)),
        ("gptneox", ("pythia-70m-q2-k-ort-genai-0.15.2",)),
        ("starcoder", ("tiny-starcoder-q2-k-ort-genai-0.15.2",)),
    ]

    pins = {pin.artifact_id for pin in MMPROJ_ARTIFACT_PINS}
    for spec in iter_projector_specs():
        if spec.runtime is Support.SUPPORTED:
            assert spec.real_artifact_ids
            assert set(spec.real_artifact_ids) <= pins


def test_all_active_stored_qtypes_have_an_import_route() -> None:
    active = [
        spec
        for spec in iter_quant_specs()
        if spec.readable and spec.role is StorageRole.QUANTIZED
    ]
    assert len(active) == 25
    routed = [
        spec
        for spec in active
        if (
            spec.native_preserve is not None
            or spec.affine_repack is not None
            or spec.dequantize is Support.SUPPORTED
        )
    ]
    assert len(routed) == 24
    assert {
        spec.name
        for spec in active
        if spec not in routed and spec.dequantize is Support.DEFERRED and spec.reason
    } == {"Q2_0"}


def test_generated_ids_are_sorted_and_unique() -> None:
    architectures = [spec.gguf_arch for spec in iter_arch_specs()]
    assert len(architectures) == len(set(architectures))
    aliases = [alias for spec in iter_arch_specs() for alias in spec.aliases]
    assert len(aliases) == len(set(aliases))
    assert not (set(architectures) & set(aliases))

    projector_ids = sorted(spec.projector_type for spec in iter_projector_specs())
    assert len(projector_ids) == len(set(projector_ids))
    tokenizer_ids = sorted(tokenizer_pre_policies())
    assert len(tokenizer_ids) == len(set(tokenizer_ids))
    blocks = render_blocks()
    assert [
        line.split("`", 2)[1]
        for line in blocks["projectors"].splitlines()
        if line.startswith("| `")
    ] == projector_ids
    assert [
        line.split("`", 2)[1]
        for line in blocks["tokenizers"].splitlines()
        if line.startswith("| `")
    ] == tokenizer_ids
