# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the block-scaled FP8 / packed-FP4 quantized weight contract.

Covers property-based classification, logical-vs-packed shape validation, scale
pairing, byte-preservation, bounded lazy loading, the byte-exact expert-major
bank stacking primitive, and the runtime emission gate's typed reject. Real
DeepSeek-V4 ``quantization_config`` + a measured slice of the checkpoint index
metadata are used as fixtures (no full-weight download); tiny synthetic packed
safetensors exercise byte-level behaviour.
"""

from __future__ import annotations

import json
import pathlib
import struct

import pytest
import torch
from safetensors.torch import save_file

from mobius.integrations._block_quant import (
    BlockQuantExportError,
    BlockQuantScheme,
    BlockQuantValidationError,
    LazyRawTensor,
    QuantizedTensorDescriptor,
    QuantKind,
    build_descriptors,
    classify_tensor,
    pair_weight_scales,
    plan_routed_expert_bank,
    read_raw_tensor_bytes,
    read_safetensors_header,
    runtime_representation_gap,
    stack_expert_bank,
    validate_descriptor,
)

# ---------------------------------------------------------------------------
# Real DeepSeek-V4 metadata fixtures (measured from the checkpoint headers).
# These are the exact quantization_config + a slice of index dtype/shape triples;
# no weight bytes are needed to classify/validate.
# ---------------------------------------------------------------------------

REAL_QUANT_CONFIG = {
    "quant_method": "fp8",
    "fmt": "e4m3",
    "scale_fmt": "ue8m0",
    "weight_block_size": [128, 128],
    "activation_scheme": "dynamic",
}
REAL_EXPERT_DTYPE = "fp4"

# name -> (dtype, shape) exactly as read from the real safetensors headers.
REAL_HEADER_SLICE: dict[str, tuple[str, tuple[int, ...]]] = {
    "layers.0.ffn.experts.0.w1.weight": ("I8", (2048, 2048)),
    "layers.0.ffn.experts.0.w1.scale": ("F8_E8M0", (2048, 128)),
    "layers.0.ffn.experts.0.w2.weight": ("I8", (4096, 1024)),
    "layers.0.ffn.experts.0.w2.scale": ("F8_E8M0", (4096, 64)),
    "layers.0.ffn.experts.0.w3.weight": ("I8", (2048, 2048)),
    "layers.0.ffn.experts.0.w3.scale": ("F8_E8M0", (2048, 128)),
    "layers.0.ffn.shared_experts.w1.weight": ("F8_E4M3", (2048, 4096)),
    "layers.0.ffn.shared_experts.w1.scale": ("F8_E8M0", (16, 32)),
    "layers.0.ffn.shared_experts.w2.weight": ("F8_E4M3", (4096, 2048)),
    "layers.0.ffn.shared_experts.w2.scale": ("F8_E8M0", (32, 16)),
    "layers.0.ffn.gate.weight": ("BF16", (256, 4096)),
    "layers.0.attn_norm.weight": ("BF16", (4096,)),
    "layers.0.attn.wq_a.weight": ("F8_E4M3", (1024, 4096)),
    "layers.0.attn.wq_a.scale": ("F8_E8M0", (8, 32)),
}

REAL_CHECKPOINT = pathlib.Path(
    "/datadisks/disk5/justinchu/onnx-genai-models/deepseek-v4-flash/checkpoint"
)


def _real_scheme() -> BlockQuantScheme:
    scheme = BlockQuantScheme.from_quantization_config(
        REAL_QUANT_CONFIG, expert_dtype=REAL_EXPERT_DTYPE
    )
    assert scheme is not None
    return scheme


# ---------------------------------------------------------------------------
# Scheme parsing
# ---------------------------------------------------------------------------


class TestBlockQuantScheme:
    def test_real_config_parses_owned(self):
        scheme = _real_scheme()
        assert scheme.quant_method == "fp8"
        assert scheme.weight_block_size == (128, 128)
        assert scheme.scale_fmt == "ue8m0"
        assert scheme.expert_dtype == "fp4"
        assert scheme.is_block_scaled_fp8
        assert scheme.has_packed_fp4_experts
        assert scheme.is_owned

    def test_per_tensor_fp8_not_owned(self):
        # No weight_block_size => ordinary per-tensor fp8 => not this contract.
        assert BlockQuantScheme.from_quantization_config({"quant_method": "fp8"}) is None

    def test_absent_config_not_owned(self):
        assert BlockQuantScheme.from_quantization_config(None) is None

    def test_fp4_experts_without_method(self):
        scheme = BlockQuantScheme.from_quantization_config(
            {"quant_method": "none"}, expert_dtype="fp4"
        )
        assert scheme is not None and scheme.has_packed_fp4_experts

    def test_from_hf_config_reads_expert_dtype(self):
        hf = type("HF", (), {})()
        hf.quantization_config = REAL_QUANT_CONFIG
        hf.expert_dtype = "fp4"
        scheme = BlockQuantScheme.from_hf_config(hf)
        assert scheme is not None and scheme.is_owned


# ---------------------------------------------------------------------------
# Property-based classification against real metadata
# ---------------------------------------------------------------------------


class TestClassifyRealMetadata:
    def test_fp4_packed_expert(self):
        scheme = _real_scheme()
        d = classify_tensor(
            "layers.0.ffn.experts.0.w1.weight",
            "I8",
            (2048, 2048),
            scale_dtype="F8_E8M0",
            scale_shape=(2048, 128),
            scale_name="layers.0.ffn.experts.0.w1.scale",
            scheme=scheme,
        )
        assert d.kind is QuantKind.FP4_PACKED
        assert d.packed_shape == (2048, 2048)
        assert d.logical_shape == (2048, 4096)  # last dim doubled (2 nibbles/byte)
        assert d.block_shape == (1, 32)
        assert d.microscale_kind == "mxfp4"
        assert d.is_routed_expert and not d.is_shared_expert
        assert d.pack_factor == 2
        validate_descriptor(d)

    def test_block_fp8_shared_expert(self):
        scheme = _real_scheme()
        d = classify_tensor(
            "layers.0.ffn.shared_experts.w1.weight",
            "F8_E4M3",
            (2048, 4096),
            scale_dtype="F8_E8M0",
            scale_shape=(16, 32),
            scale_name="layers.0.ffn.shared_experts.w1.scale",
            scheme=scheme,
        )
        assert d.kind is QuantKind.BLOCK_FP8
        assert d.logical_shape == d.packed_shape == (2048, 4096)
        assert d.block_shape == (128, 128)
        assert d.is_shared_expert and not d.is_routed_expert
        assert d.pack_factor == 1
        validate_descriptor(d)

    def test_block_fp8_attention_projection(self):
        scheme = _real_scheme()
        d = classify_tensor(
            "layers.0.attn.wq_a.weight",
            "F8_E4M3",
            (1024, 4096),
            scale_dtype="F8_E8M0",
            scale_shape=(8, 32),
            scheme=scheme,
        )
        assert d.kind is QuantKind.BLOCK_FP8
        assert not d.is_routed_expert and not d.is_shared_expert
        validate_descriptor(d)

    def test_ordinary_router_and_norm(self):
        scheme = _real_scheme()
        gate = classify_tensor("layers.0.ffn.gate.weight", "BF16", (256, 4096), scheme=scheme)
        norm = classify_tensor("layers.0.attn_norm.weight", "BF16", (4096,), scheme=scheme)
        assert gate.kind is QuantKind.ORDINARY and gate.scale_name is None
        assert norm.kind is QuantKind.ORDINARY
        validate_descriptor(gate)
        validate_descriptor(norm)

    def test_build_descriptors_over_real_slice(self):
        scheme = _real_scheme()
        descs = build_descriptors(REAL_HEADER_SLICE, scheme)
        # Only .weight keys become descriptors; scales are consumed as pairs.
        assert all(name.endswith(".weight") for name in descs)
        kinds = {name: d.kind for name, d in descs.items()}
        assert kinds["layers.0.ffn.experts.0.w2.weight"] is QuantKind.FP4_PACKED
        assert kinds["layers.0.ffn.shared_experts.w2.weight"] is QuantKind.BLOCK_FP8
        assert kinds["layers.0.ffn.gate.weight"] is QuantKind.ORDINARY
        # Routed vs shared classification is structural.
        assert descs["layers.0.ffn.experts.0.w1.weight"].is_routed_expert
        assert descs["layers.0.ffn.shared_experts.w1.weight"].is_shared_expert


# ---------------------------------------------------------------------------
# Unsupported / malformed inputs fail closed
# ---------------------------------------------------------------------------


class TestUnsupportedAndValidation:
    def test_i8_without_scale_is_unsupported(self):
        d = classify_tensor("w.weight", "I8", (4, 8))
        assert d.kind is QuantKind.UNSUPPORTED
        assert "micro-scale" in d.unsupported_reason
        with pytest.raises(BlockQuantValidationError):
            validate_descriptor(d)

    def test_fp8_weight_without_scale_is_unsupported(self):
        d = classify_tensor("w.weight", "F8_E4M3", (4, 8))
        assert d.kind is QuantKind.UNSUPPORTED

    def test_unknown_dtype_pair_is_unsupported(self):
        d = classify_tensor(
            "w.weight", "I16", (4, 8), scale_dtype="F8_E8M0", scale_shape=(4, 1)
        )
        assert d.kind is QuantKind.UNSUPPORTED

    def test_wrong_fp4_scale_shape_raises(self):
        # logical_in = 16 -> expected scale last dim 16/32 is invalid; use a
        # deliberately wrong scale grid.
        d = classify_tensor(
            "e.experts.0.w1.weight",
            "I8",
            (8, 32),  # logical (8, 64)
            scale_dtype="F8_E8M0",
            scale_shape=(8, 3),  # wrong: expected (8, 2)
        )
        assert d.kind is QuantKind.FP4_PACKED
        with pytest.raises(BlockQuantValidationError, match="scale shape"):
            validate_descriptor(d)

    def test_wrong_block_fp8_scale_grid_raises(self):
        scheme = BlockQuantScheme.from_quantization_config(
            {"quant_method": "fp8", "weight_block_size": [128, 128]}
        )
        d = classify_tensor(
            "attn.wq_a.weight",
            "F8_E4M3",
            (1024, 4096),
            scale_dtype="F8_E8M0",
            scale_shape=(9, 32),  # wrong: expected (8, 32)
            scheme=scheme,
        )
        with pytest.raises(BlockQuantValidationError, match="scale grid"):
            validate_descriptor(d)

    def test_ordinary_must_not_carry_scale(self):
        d = QuantizedTensorDescriptor(
            name="x.weight",
            kind=QuantKind.ORDINARY,
            weight_dtype="BF16",
            logical_shape=(4, 4),
            packed_shape=(4, 4),
            weight_num_bytes=32,
            is_routed_expert=False,
            is_shared_expert=False,
            scale_name="x.scale",
            scale_shape=(1, 1),
        )
        with pytest.raises(BlockQuantValidationError):
            validate_descriptor(d)


# ---------------------------------------------------------------------------
# Scale pairing: missing / duplicate / orphan
# ---------------------------------------------------------------------------


class TestScalePairing:
    def test_pairs_weight_with_scale(self):
        index = {
            "a.weight": ("F8_E4M3", (4, 4)),
            "a.scale": ("F8_E8M0", (1, 1)),
            "b.weight": ("BF16", (2, 2)),
        }
        pairing = pair_weight_scales(index)
        assert pairing == {"a.weight": "a.scale", "b.weight": None}

    def test_orphan_scale_raises(self):
        index = {
            "a.weight": ("F8_E4M3", (4, 4)),
            "a.scale": ("F8_E8M0", (1, 1)),
            "ghost.scale": ("F8_E8M0", (1, 1)),  # no matching .weight
        }
        with pytest.raises(BlockQuantValidationError, match="orphan"):
            pair_weight_scales(index)

    def test_missing_scale_for_quantized_weight_raises_on_validate(self):
        # A block-fp8 weight whose scale is absent classifies UNSUPPORTED and
        # cannot be validated as a real tensor.
        index = {"a.weight": ("F8_E4M3", (4, 4))}
        descs = build_descriptors(index, _real_scheme(), validate=False)
        assert descs["a.weight"].kind is QuantKind.UNSUPPORTED
        with pytest.raises(BlockQuantValidationError):
            validate_descriptor(descs["a.weight"])


# ---------------------------------------------------------------------------
# Byte-preserving raw reader + bounded lazy loader (synthetic safetensors)
# ---------------------------------------------------------------------------


def _write_synthetic_experts(directory: pathlib.Path, n_experts: int = 3) -> pathlib.Path:
    """Write a tiny safetensors file with fp4-packed experts + real dtypes."""
    state: dict[str, torch.Tensor] = {}
    torch.manual_seed(0)
    for e in range(n_experts):
        # logical (8, 64) -> packed int8 (8, 32); scale e8m0 (8, 2).
        state[f"layers.0.ffn.experts.{e}.w1.weight"] = torch.randint(
            -128, 127, (8, 32), dtype=torch.int8
        )
        state[f"layers.0.ffn.experts.{e}.w1.scale"] = torch.randint(
            0, 254, (8, 2), dtype=torch.uint8
        ).view(torch.float8_e8m0fnu)
    # One block-fp8 shared projection with real E4M3 storage.
    state["layers.0.ffn.shared_experts.w1.weight"] = torch.randint(
        0, 200, (4, 4), dtype=torch.uint8
    ).view(torch.float8_e4m3fn)
    state["layers.0.ffn.shared_experts.w1.scale"] = torch.randint(
        0, 254, (2, 2), dtype=torch.uint8
    ).view(torch.float8_e8m0fnu)
    path = directory / "model.safetensors"
    save_file(state, str(path))
    return path


class TestRawReaderAndLazy:
    def test_raw_bytes_are_byte_exact(self, tmp_path):
        path = _write_synthetic_experts(tmp_path)
        key = "layers.0.ffn.experts.0.w1.weight"
        raw = read_raw_tensor_bytes(path, key)
        # Compare against the tensor's own byte view (uint8) — must be identical.
        from safetensors import safe_open

        with safe_open(path, framework="pt") as h:
            t = h.get_tensor(key)
        assert raw == t.view(torch.uint8).contiguous().numpy().tobytes()

    def test_lazy_tensor_knows_size_from_header(self, tmp_path):
        path = _write_synthetic_experts(tmp_path)
        lazy = LazyRawTensor.open(path, "layers.0.ffn.experts.0.w1.weight")
        assert lazy.dtype == "I8"
        assert lazy.shape == (8, 32)
        assert lazy.num_bytes == 8 * 32  # 1 byte/int8
        # Reading is deferred and repeatable (byte-exact each time).
        first = lazy.read()
        assert len(first) == lazy.num_bytes
        assert lazy.read() == first

    def test_e8m0_scale_bytes_preserved(self, tmp_path):
        path = _write_synthetic_experts(tmp_path)
        raw = read_raw_tensor_bytes(path, "layers.0.ffn.experts.0.w1.scale")
        assert len(raw) == 8 * 2  # e8m0 is 1 byte/elem

    def test_header_only_reader_does_not_scan_data(self, tmp_path):
        path = _write_synthetic_experts(tmp_path)
        header = read_safetensors_header(path)
        assert "layers.0.ffn.experts.0.w1.weight" in header
        assert header["layers.0.ffn.experts.0.w1.weight"]["dtype"] == "I8"


# ---------------------------------------------------------------------------
# Byte-exact expert-major bank stacking
# ---------------------------------------------------------------------------


class TestExpertBankStacking:
    def test_stack_is_byte_exact_and_recoverable(self, tmp_path):
        path = _write_synthetic_experts(tmp_path, n_experts=4)
        per_expert = [
            read_raw_tensor_bytes(path, f"layers.0.ffn.experts.{e}.w1.weight")
            for e in range(4)
        ]
        bank = stack_expert_bank(
            per_expert, per_expert_packed_shape=(8, 32), weight_dtype="I8"
        )
        assert bank.num_experts == 4
        assert bank.per_expert_num_bytes == 8 * 32
        assert bank.data == b"".join(per_expert)
        for e in range(4):
            assert bank.expert_bytes(e) == per_expert[e]

    def test_ragged_bank_raises(self):
        with pytest.raises(BlockQuantValidationError, match="ragged"):
            stack_expert_bank(
                [b"\x00\x01", b"\x02"], per_expert_packed_shape=(1, 2), weight_dtype="I8"
            )

    def test_empty_bank_raises(self):
        with pytest.raises(BlockQuantValidationError):
            stack_expert_bank([], per_expert_packed_shape=(1, 2), weight_dtype="I8")


# ---------------------------------------------------------------------------
# Runtime emission gate: typed reject with exact ABI gap
# ---------------------------------------------------------------------------


def _routed_fp4_descs(n: int = 3) -> list[QuantizedTensorDescriptor]:
    scheme = _real_scheme()
    return [
        classify_tensor(
            f"layers.0.ffn.experts.{e}.w1.weight",
            "I8",
            (2048, 2048),
            scale_dtype="F8_E8M0",
            scale_shape=(2048, 128),
            scheme=scheme,
        )
        for e in range(n)
    ]


class TestEmissionGate:
    def test_gap_none_for_ordinary(self):
        d = classify_tensor("gate.weight", "BF16", (8, 8))
        assert runtime_representation_gap(d) is None

    def test_gap_block_fp8_names_missing_format(self):
        scheme = _real_scheme()
        d = classify_tensor(
            "attn.wq_a.weight",
            "F8_E4M3",
            (1024, 4096),
            scale_dtype="F8_E8M0",
            scale_shape=(8, 32),
            scheme=scheme,
        )
        gap = runtime_representation_gap(d)
        assert gap is not None and "block-FP8" in gap

    def test_gap_fp4_names_planar_vs_interleaved(self):
        d = _routed_fp4_descs(1)[0]
        gap = runtime_representation_gap(d)
        assert gap is not None
        assert "planar" in gap and "block_mxfp4" in gap

    def test_plan_routed_bank_typed_rejects_fp4(self):
        with pytest.raises(BlockQuantExportError) as ei:
            plan_routed_expert_bank(_routed_fp4_descs(3))
        msg = str(ei.value)
        assert "not representable" in msg
        assert "no dense fallback" in msg

    def test_plan_rejects_mixed_bank(self):
        scheme = _real_scheme()
        experts = _routed_fp4_descs(2)
        odd = classify_tensor(
            "layers.0.ffn.experts.2.w2.weight",
            "I8",
            (4096, 1024),  # different packed shape
            scale_dtype="F8_E8M0",
            scale_shape=(4096, 64),
            scheme=scheme,
        )
        with pytest.raises(BlockQuantValidationError, match="mixed expert bank"):
            plan_routed_expert_bank([*experts, odd])

    def test_plan_rejects_non_routed(self):
        scheme = _real_scheme()
        shared = classify_tensor(
            "layers.0.ffn.shared_experts.w1.weight",
            "F8_E4M3",
            (2048, 4096),
            scale_dtype="F8_E8M0",
            scale_shape=(16, 32),
            scheme=scheme,
        )
        with pytest.raises(BlockQuantValidationError, match="non-routed"):
            plan_routed_expert_bank([shared])

    def test_plan_empty_raises(self):
        with pytest.raises(BlockQuantValidationError):
            plan_routed_expert_bank([])


# ---------------------------------------------------------------------------
# Integration with QuantizationConfig.from_transformers (typed blocker)
# ---------------------------------------------------------------------------


class TestFromTransformersBlocker:
    def _hf(self, **kw):
        o = type("HF", (), {})()
        for k, v in kw.items():
            setattr(o, k, v)
        return o

    def test_block_scaled_fp8_raises_typed(self):
        from mobius._configs import QuantizationConfig

        hf = self._hf(quantization_config=REAL_QUANT_CONFIG, expert_dtype="fp4")
        with pytest.raises(BlockQuantExportError, match="Block-scaled FP8"):
            QuantizationConfig.from_transformers(hf)

    def test_per_tensor_fp8_still_returns_none(self):
        from mobius._configs import QuantizationConfig

        hf = self._hf(quantization_config={"quant_method": "fp8", "bits": 8})
        assert QuantizationConfig.from_transformers(hf) is None

    def test_gptq_unaffected(self):
        from mobius._configs import QuantizationConfig

        hf = self._hf(quantization_config={"quant_method": "gptq", "bits": 4})
        assert QuantizationConfig.from_transformers(hf).quant_method == "gptq"


# ---------------------------------------------------------------------------
# Opt-in: exercise the real checkpoint headers when the disk is present.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not REAL_CHECKPOINT.is_dir(), reason="real DeepSeek-V4 checkpoint not mounted"
)
class TestRealCheckpointHeaders:
    def _index(self) -> dict:
        return json.loads((REAL_CHECKPOINT / "model.safetensors.index.json").read_text())

    def test_real_layer0_classifies_and_gate_rejects(self):
        scheme = _real_scheme()
        wm = self._index()["weight_map"]
        # Build a header index for layer 0 by reading only shard headers.
        keys = [
            k
            for k in wm
            if k.startswith("layers.0.")
            and (".experts.0." in k or "shared_experts" in k or k.endswith("gate.weight"))
        ]
        header_index: dict[str, tuple[str, tuple[int, ...]]] = {}
        hdr_cache: dict[str, dict] = {}
        for k in keys:
            shard = wm[k]
            if shard not in hdr_cache:
                hdr_cache[shard] = read_safetensors_header(REAL_CHECKPOINT / shard)
            e = hdr_cache[shard][k]
            header_index[k] = (e["dtype"], tuple(e["shape"]))
        descs = build_descriptors(header_index, scheme)
        routed = [
            d for d in descs.values() if d.is_routed_expert and d.name.endswith("w1.weight")
        ]
        assert routed and all(d.kind is QuantKind.FP4_PACKED for d in routed)
        with pytest.raises(BlockQuantExportError):
            plan_routed_expert_bank(routed)

    def test_real_expert_bytes_preserved(self):
        wm = self._index()["weight_map"]
        key = "layers.0.ffn.experts.0.w1.weight"
        shard = REAL_CHECKPOINT / wm[key]
        raw = read_raw_tensor_bytes(shard, key)
        _dtype, shape, start, end = _span(shard, key)
        assert len(raw) == end - start
        assert shape == (2048, 2048)


def _span(path, key):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    entry = header[key]
    base = 8 + n
    s, e = entry["data_offsets"]
    return entry["dtype"], tuple(entry["shape"]), base + s, base + e
