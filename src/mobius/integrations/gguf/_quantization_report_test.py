# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import json

import pytest

from mobius.integrations.gguf._quantization_report import (
    GGUFQuantizationReport,
    QuantizationDisposition,
    QuantizationTensorRecord,
)


def _record(
    name: str,
    qtype: str,
    source_bytes: int,
    disposition: QuantizationDisposition,
    target: str,
) -> QuantizationTensorRecord:
    return QuantizationTensorRecord(
        name=name,
        qtype=qtype,
        source_bytes=source_bytes,
        disposition=disposition,
        target_storage=target,
        reason="tested route",
    )


def test_report_is_deterministic_and_round_trips(tmp_path) -> None:
    report = GGUFQuantizationReport.create(
        source_qtypes=[
            ("Q6_K", 210),
            ("F32", 128),
            ("Q4_K", 144),
            ("Q5_K", 176),
        ],
        tensor_records=[
            _record(
                "blk.0.attn_v.weight",
                "Q6_K",
                210,
                QuantizationDisposition.LOSSY_REQUANTIZE,
                "INT4 affine block-32",
            ),
            _record(
                "blk.0.attn_q.weight",
                "Q4_K",
                144,
                QuantizationDisposition.LOSSY_REQUANTIZE,
                "INT4 affine block-32",
            ),
            _record(
                "token_embd.weight",
                "Q5_K",
                176,
                QuantizationDisposition.LOSSY_REQUANTIZE,
                "INT4 affine block-32",
            ),
            _record(
                "output_norm.weight",
                "F32",
                128,
                QuantizationDisposition.SOURCE_FLOAT,
                "float",
            ),
        ],
        target_storage_format="INT4 affine block-32",
        compute_mode="runtime-dependent native custom op or inline standard-ONNX fallback",
        compute_capability="no kernel promise",
    )
    path = tmp_path / "quantization_report.json"
    report.write_json(path)

    assert GGUFQuantizationReport.read_json(path) == report
    assert report.converted_from == "Q4_K_M-like mixed GGUF"
    assert report.storage_quantized is True
    assert report.source_fidelity is False
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert [stat["qtype"] for stat in payload["source_qtype_census"]] == [
        "F32",
        "Q4_K",
        "Q5_K",
        "Q6_K",
    ]


def test_converted_from_is_census_based_not_filename() -> None:
    report = GGUFQuantizationReport.create(
        source_qtypes=[("Q4_K", 144)],
        tensor_records=[
            _record(
                "a-Q4_K_M.gguf.weight",
                "Q4_K",
                144,
                QuantizationDisposition.LOSSY_REQUANTIZE,
                "INT4 affine block-32",
            )
        ],
        target_storage_format="INT4 affine block-32",
        compute_mode="custom op",
        compute_capability="no kernel promise",
    )

    assert report.converted_from is None


def test_malformed_record_fails_closed(tmp_path) -> None:
    path = tmp_path / "quantization_report.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "converted_from": None,
                "source_qtype_census": [],
                "target_storage_format": "float",
                "dispositions": [],
                "source_fidelity": True,
                "storage_quantized": False,
                "compute_mode": "float",
                "compute_capability": "float",
                "explicit_float_tensors": [],
                "tensor_records": ["not an object"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="entries must be objects"):
        GGUFQuantizationReport.read_json(path)


def test_inconsistent_fidelity_claim_fails_closed(tmp_path) -> None:
    report = GGUFQuantizationReport.create(
        source_qtypes=[("Q5_K", 176)],
        tensor_records=[
            _record(
                "token_embd.weight",
                "Q5_K",
                176,
                QuantizationDisposition.LOSSY_REQUANTIZE,
                "INT4 affine block-32",
            )
        ],
        target_storage_format="INT4 affine block-32",
        compute_mode="custom op or fallback",
        compute_capability="no kernel promise",
    )
    payload = report.to_dict()
    payload["source_fidelity"] = True
    path = tmp_path / "quantization_report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="inconsistent with tensor records"):
        GGUFQuantizationReport.read_json(path)


def test_component_reports_combine_without_losing_float_records() -> None:
    common = {
        "source_qtypes": [("Q4_0", 144), ("Q4_1", 160)],
        "target_storage_format": "INT4 affine block-32",
        "compute_mode": "custom op or fallback",
        "compute_capability": "no kernel promise",
    }
    backbone = GGUFQuantizationReport.create(
        **common,
        tensor_records=[
            _record(
                "blk.0.attn_q.weight",
                "Q4_0",
                144,
                QuantizationDisposition.LOSSLESS_REPACK,
                "INT4 affine block-32",
            )
        ],
    )
    sidecar = GGUFQuantizationReport.create(
        **common,
        tensor_records=[
            _record(
                "blk.1.nextn.shared_head_head.weight",
                "Q4_1",
                160,
                QuantizationDisposition.DEQUANTIZED_FLOAT,
                "float",
            )
        ],
    )

    combined = GGUFQuantizationReport.combine(backbone, sidecar)

    assert combined.source_fidelity is False
    assert combined.storage_quantized is True
    assert [record.name for record in combined.explicit_float_tensors] == [
        "blk.1.nextn.shared_head_head.weight"
    ]
