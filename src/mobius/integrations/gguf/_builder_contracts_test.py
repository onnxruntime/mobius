# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for GGUF preflight, cache, raw tensor, and weight contracts."""

from __future__ import annotations

import re
import struct
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpx
import numpy as np
import pytest
from huggingface_hub.utils import OfflineModeIsEnabled

from mobius.integrations.gguf._builder_test_utils import (
    _gguf_header_prefix,
    _write_falcon_h1_gguf,
    _write_quantized_gguf,
)
from mobius.integrations.gguf._builder_test_utils import (
    q4_0_gguf as q4_0_gguf,
)


class TestBuildGgufStaticCache:
    """Tests for build_from_gguf(static_cache=True).

    Static cache mode replaces the dynamic concat-grow KV cache with
    pre-allocated fixed-width buffers (written in place via TensorScatter),
    producing a fully static-shaped graph required by fixed-shape runtimes
    such as the QNN HTP backend. Llama uses the base ``DecoderLayer`` which
    supports the StaticCacheState dispatch.
    """

    def test_static_cache_emits_fixed_width_cache_io(self, q4_0_gguf: Path):
        """Static cache build exposes fixed-width key_cache/value_cache I/O."""
        from mobius.integrations.gguf import build_from_gguf

        max_seq_len = 128
        model = build_from_gguf(
            q4_0_gguf, keep_quantized=True, static_cache=True, max_seq_len=max_seq_len
        )["model"]

        input_names = {i.name for i in model.graph.inputs}
        # Static cache uses key_cache.N / value_cache.N inputs, not the
        # dynamic past_key_values.N.key / .value pair.
        assert any(n and n.startswith("key_cache.") for n in input_names), (
            f"Expected key_cache.* inputs, got {sorted(input_names)}"
        )
        assert not any(n and n.startswith("past_key_values.") for n in input_names), (
            f"Static cache must not emit past_key_values.* inputs, got {sorted(input_names)}"
        )

        # The KV axis of every cache buffer must be a concrete int == max_seq_len,
        # i.e. fully static (no symbolic dims).
        for inp in model.graph.inputs:
            name = inp.name or ""
            if name.startswith(("key_cache.", "value_cache.")):
                assert inp.shape is not None
                assert inp.shape[1] == max_seq_len, (
                    f"{name} KV axis {inp.shape[1]!r} != max_seq_len {max_seq_len}"
                )

    def test_static_cache_rejects_explicit_task(self, q4_0_gguf: Path):
        """static_cache=True with an explicit task override is a ValueError."""
        from mobius.integrations.gguf import build_from_gguf

        with pytest.raises(ValueError, match="static_cache"):
            build_from_gguf(
                q4_0_gguf,
                keep_quantized=True,
                static_cache=True,
                task="text-generation",
            )


class TestMultimodalQuantizationDefault:
    @pytest.mark.parametrize("keep_quantized", [True, False])
    def test_build_from_gguf_forwards_quantization_policy_to_mmproj(
        self, keep_quantized: bool
    ):
        from mobius.integrations.gguf import build_from_gguf
        from mobius.integrations.gguf._tokenizer import GGUFTokenizerVerdict

        expected = mock.MagicMock()
        expected.__iter__.return_value = iter(("model", "vision"))
        expected.config = SimpleNamespace(model_type="test-vlm")
        expected.gguf_source_path = "text.gguf"
        expected.gguf_source_filename = "text.gguf"
        expected.gguf_tokenizer_verdict = GGUFTokenizerVerdict(
            route="copy",
            model="gpt2",
            pre="gpt-2",
            canonical_pre="gpt-2",
            reason="exact embedded tokenizer",
            token_count=2,
        )
        expected.export_report = None
        text_model = SimpleNamespace(
            architecture="llama",
            metadata={},
            source_identity=(1, 2, 3),
            source_matches_path=lambda: True,
        )
        with mock.patch(
            "mobius.integrations.gguf._mmproj.build_vlm_from_gguf",
            return_value=expected,
        ) as build_multimodal:
            if keep_quantized:
                actual = build_from_gguf(
                    "text.gguf",
                    mmproj="mmproj.gguf",
                    image_token_id=-200,
                    _gguf_model=text_model,
                )
            else:
                actual = build_from_gguf(
                    "text.gguf",
                    mmproj="mmproj.gguf",
                    image_token_id=-200,
                    keep_quantized=False,
                    _gguf_model=text_model,
                )

        assert actual is expected
        build_multimodal.assert_called_once_with(
            "text.gguf",
            "mmproj.gguf",
            dtype=None,
            execution_provider="default",
            image_token_id=-200,
            keep_quantized=keep_quantized,
            _text_gguf_model=text_model,
        )

    def test_image_token_id_requires_mmproj(self):
        from mobius.integrations.gguf import build_from_gguf

        with pytest.raises(ValueError, match="requires a companion mmproj"):
            build_from_gguf("text.gguf", image_token_id=-200)


class TestRawTensorIterator:
    """Tests for GGUFModel.tensor_items_raw()."""

    def test_yields_raw_data(self, q4_0_gguf: Path):
        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf._reader import GGUFModel

        model = GGUFModel(q4_0_gguf)
        items = list(model.tensor_items_raw())

        # Should have tensors
        assert len(items) > 0

        # Check a quantized tensor
        q_items = [(n, d, qt, s) for n, d, qt, s in items if qt == GGMLQuantizationType.Q4_0]
        assert len(q_items) > 0
        _name, raw, _qtype, shape = q_items[0]
        assert raw.dtype == np.uint8
        assert len(shape) == 2

    def test_float_tensors_have_correct_type(self, q4_0_gguf: Path):
        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf._reader import GGUFModel

        model = GGUFModel(q4_0_gguf)

        f32_items = [
            (n, d, qt, s)
            for n, d, qt, s in model.tensor_items_raw()
            if qt == GGMLQuantizationType.F32
        ]
        assert len(f32_items) > 0
        for _name, _raw, qtype, _shape in f32_items:
            assert qtype == GGMLQuantizationType.F32

    def test_dequantize_raw_tensor_matches_get_tensor(self, q4_0_gguf: Path):
        from mobius.integrations.gguf._reader import GGUFModel

        model = GGUFModel(q4_0_gguf)
        name, raw, qtype, shape = next(
            item for item in model.tensor_items_raw() if len(item[3]) == 2
        )
        expected = model.get_tensor(name)
        actual = model.dequantize_raw_tensor(raw, qtype, shape)
        np.testing.assert_array_equal(actual, expected)


class TestGGUFPreflightGuards:
    """Unsupported layouts fail before graph construction or large downloads."""

    def test_nemotron_layout_excludes_combined_mtp_block(self):
        from mobius.integrations.gguf._builder import (
            _summarize_nemotron_h_moe_layout,
        )

        # Pinned NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 schedule:
        # 52 backbone layers followed by one combined attention+MoE MTP block.
        backbone_schedule = (
            "mamba",
            "moe",
            "mamba",
            "moe",
            "mamba",
            "attention",
            "moe",
            "mamba",
            "moe",
            "mamba",
            "moe",
            "mamba",
            "attention",
            "moe",
            "mamba",
            "moe",
            "mamba",
            "moe",
            "mamba",
            "attention",
            "moe",
            "mamba",
            "moe",
            "mamba",
            "moe",
            "mamba",
            "attention",
            "moe",
            "mamba",
            "moe",
            "mamba",
            "moe",
            "mamba",
            "attention",
            "moe",
            "mamba",
            "moe",
            "mamba",
            "moe",
            "mamba",
            "moe",
            "mamba",
            "attention",
            "moe",
            "mamba",
            "moe",
            "mamba",
            "moe",
            "mamba",
            "moe",
            "mamba",
            "moe",
        )
        assert len(backbone_schedule) == 52

        representative_tensor = {
            "mamba": "ssm_in.weight",
            "moe": "ffn_up_exps.weight",
            "attention": "attn_q.weight",
        }
        tensor_names = [
            f"blk.{index}.{representative_tensor[layer_type]}"
            for index, layer_type in enumerate(backbone_schedule)
        ]
        tensor_names.extend(
            [
                "blk.52.nextn.eh_proj.weight",
                "blk.52.attn_q.weight",
                "blk.52.ffn_up_exps.weight",
            ]
        )

        counts, mtp_blocks, mtp_kinds = _summarize_nemotron_h_moe_layout(tensor_names)

        assert dict(counts) == {"mamba": 23, "moe": 23, "attention": 6}
        assert mtp_blocks == (52,)
        assert mtp_kinds == {52: frozenset({"attention", "moe"})}

    @pytest.mark.parametrize(
        ("architecture", "projection_quantization"),
        [
            pytest.param(architecture, quantization, id=f"{architecture}-{quantization}")
            for architecture in ("pockettts", "qwen3tts", "wavtokenizer-dec")
            for quantization in ("f32", "q4_0")
        ],
    )
    @pytest.mark.parametrize(
        "build_options",
        [
            pytest.param({}, id="default"),
            pytest.param({"dtype": "f16"}, id="dtype"),
            pytest.param({"static_cache": True}, id="static-cache"),
            pytest.param({"keep_quantized": False}, id="dequantize"),
        ],
    )
    def test_deferred_audio_architectures_fail_before_all_downstream_stages(
        self,
        architecture: str,
        projection_quantization: str,
        build_options: dict[str, object],
        tmp_path: Path,
    ) -> None:
        from mobius import _builder as core_builder
        from mobius import _registry as core_registry
        from mobius.integrations.gguf import _builder as gguf_builder
        from mobius.integrations.gguf import _config_mapping, build_from_gguf
        from mobius.integrations.gguf._errors import UnsupportedGGUFArchitectureError

        path = tmp_path / f"{architecture}-{projection_quantization}.gguf"
        _write_quantized_gguf(
            path,
            architecture=architecture,
            projection_quantization=projection_quantization,
        )

        downstream = AssertionError("deferred architecture reached a downstream stage")
        with (
            mock.patch.object(
                gguf_builder, "_has_quantized_weights", side_effect=downstream
            ) as quantization_probe,
            mock.patch.object(
                _config_mapping, "gguf_to_config", side_effect=downstream
            ) as config_extraction,
            mock.patch.object(
                core_registry.registry, "get", side_effect=downstream
            ) as module_lookup,
            mock.patch.object(
                core_builder, "build_from_module", side_effect=downstream
            ) as graph_build,
            pytest.raises(
                UnsupportedGGUFArchitectureError,
                match=rf"{re.escape(architecture)}.*before config extraction",
            ),
        ):
            build_from_gguf(path, **build_options)

        quantization_probe.assert_not_called()
        config_extraction.assert_not_called()
        module_lookup.assert_not_called()
        graph_build.assert_not_called()

    def test_invalid_decay_fails_before_graph(self, tmp_path: Path, monkeypatch) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "falcon-h1-invalid-decay.gguf"
        _write_falcon_h1_gguf(path, quantized=False, invalid_decay=True)
        graph_build = mock.Mock(side_effect=AssertionError("graph construction reached"))
        monkeypatch.setattr(core_builder, "build_from_module", graph_build)

        with pytest.raises(ValueError, match="finite negative"):
            build_from_gguf(path)
        graph_build.assert_not_called()

    @pytest.mark.parametrize(
        "architecture",
        [
            "bailingmoe3",
            "deepseek4",
            "gpt-oss",
            "chameleon",
            "cogvlm",
            "deepseek2-ocr",
            "gemma3n",
            "hunyuan_vl",
            "llama4",
            "mistral3",
            "paddleocr",
            "qwen3vl",
            "qwen3vlmoe",
        ],
    )
    def test_deferred_audited_cohorts_fail_before_graph_construction(
        self, architecture: str, tmp_path: Path
    ) -> None:
        from mobius import _builder as core_builder
        from mobius import _registry as core_registry
        from mobius.integrations.gguf import _builder as gguf_builder
        from mobius.integrations.gguf import _config_mapping, build_from_gguf
        from mobius.integrations.gguf._errors import UnsupportedGGUFArchitectureError

        path = tmp_path / f"{architecture}.gguf"
        _write_quantized_gguf(path, architecture=architecture)
        downstream = AssertionError("deferred architecture reached graph construction")
        with (
            mock.patch.object(
                gguf_builder, "_has_quantized_weights", side_effect=downstream
            ) as quantization_probe,
            mock.patch.object(
                _config_mapping, "gguf_to_config", side_effect=downstream
            ) as config_extraction,
            mock.patch.object(
                core_registry.registry, "get", side_effect=downstream
            ) as module_lookup,
            mock.patch.object(
                core_builder, "build_from_module", side_effect=downstream
            ) as graph_build,
            pytest.raises(UnsupportedGGUFArchitectureError, match=architecture),
        ):
            build_from_gguf(path)

        quantization_probe.assert_not_called()
        config_extraction.assert_not_called()
        module_lookup.assert_not_called()
        graph_build.assert_not_called()

    @pytest.mark.parametrize("projection_quantization", ["f32", "q4_0"])
    def test_standalone_clip_fails_before_all_downstream_stages(
        self, projection_quantization: str, tmp_path: Path
    ) -> None:
        from mobius import _builder as core_builder
        from mobius import _registry as core_registry
        from mobius.integrations.gguf import _builder as gguf_builder
        from mobius.integrations.gguf import _config_mapping, build_from_gguf
        from mobius.integrations.gguf._errors import DisabledGGUFArchitectureError

        path = tmp_path / f"standalone-clip-{projection_quantization}.gguf"
        _write_quantized_gguf(
            path,
            architecture="clip",
            projection_quantization=projection_quantization,
        )

        downstream = AssertionError("standalone clip reached a downstream stage")
        with (
            mock.patch.object(
                gguf_builder, "_has_quantized_weights", side_effect=downstream
            ) as quantization_probe,
            mock.patch.object(
                _config_mapping, "gguf_to_config", side_effect=downstream
            ) as config_extraction,
            mock.patch.object(
                core_registry.registry, "get", side_effect=downstream
            ) as module_lookup,
            mock.patch.object(
                core_builder, "build_from_module", side_effect=downstream
            ) as graph_build,
            pytest.raises(
                DisabledGGUFArchitectureError,
                match=r"clip.*intentionally disabled",
            ),
        ):
            build_from_gguf(path)

        quantization_probe.assert_not_called()
        config_extraction.assert_not_called()
        module_lookup.assert_not_called()
        graph_build.assert_not_called()

    def test_remote_deferred_audio_architecture_fails_before_download(self):
        from mobius.integrations.gguf._builder import _resolve_gguf_path
        from mobius.integrations.gguf._errors import UnsupportedGGUFArchitectureError

        filename = "talkie-f16.gguf"
        with (
            mock.patch(
                "mobius.integrations.gguf._builder._preflight_hf_gguf_file",
                side_effect=UnsupportedGGUFArchitectureError(
                    "talkie before config extraction"
                ),
            ),
            mock.patch("mobius.integrations.gguf._builder.hf_hub_download") as download,
            pytest.raises(
                UnsupportedGGUFArchitectureError,
                match=r"talkie.*before config extraction",
            ),
        ):
            _resolve_gguf_path(f"example/talkie:{filename}")

        download.assert_not_called()

    def test_remote_standalone_clip_fails_before_download(self):
        from mobius.integrations.gguf._builder import _resolve_gguf_path
        from mobius.integrations.gguf._errors import DisabledGGUFArchitectureError

        filename = "mmproj-f16.gguf"
        with (
            mock.patch(
                "mobius.integrations.gguf._builder._preflight_hf_gguf_file",
                side_effect=DisabledGGUFArchitectureError("clip intentionally disabled"),
            ),
            mock.patch("mobius.integrations.gguf._builder.hf_hub_download") as download,
            pytest.raises(
                DisabledGGUFArchitectureError,
                match=r"clip.*intentionally disabled",
            ),
        ):
            _resolve_gguf_path(f"example/mmproj:{filename}")

        download.assert_not_called()

    def test_exact_selected_clip_header_returns_immutable_revision(self) -> None:
        from mobius.integrations.gguf._builder import (
            _GGUF_HEADER_RANGE_BYTES,
            _preflight_hf_mmproj_companion_file,
        )

        response = mock.MagicMock()
        response.iter_bytes.return_value = [_gguf_header_prefix("clip")]
        response_context = mock.MagicMock()
        response_context.__enter__.return_value = response
        session = mock.MagicMock()
        session.stream.return_value = response_context
        commit_hash = "c" * 40

        with (
            mock.patch(
                "mobius.integrations.gguf._builder.hf_hub_url",
                return_value="https://huggingface.co/exact-file",
            ) as hub_url,
            mock.patch(
                "mobius.integrations.gguf._builder.get_hf_file_metadata",
                return_value=SimpleNamespace(
                    commit_hash=commit_hash,
                    location="https://huggingface.co/mmproj.gguf",
                ),
            ) as file_metadata,
            mock.patch(
                "mobius.integrations.gguf._builder.get_session",
                return_value=session,
            ),
            mock.patch(
                "mobius.integrations.gguf._builder.build_hf_headers",
                return_value={"authorization": "Bearer test-token"},
            ),
        ):
            actual = _preflight_hf_mmproj_companion_file(
                "example/gemma4",
                "nested/mmproj-F16.gguf",
                revision="pinned-revision",
            )

        assert actual == commit_hash
        hub_url.assert_called_once_with(
            "example/gemma4",
            "nested/mmproj-F16.gguf",
            revision="pinned-revision",
        )
        file_metadata.assert_called_once_with("https://huggingface.co/exact-file")
        session.stream.assert_called_once()
        stream_args = session.stream.call_args
        assert stream_args.args == ("GET", "https://huggingface.co/mmproj.gguf")
        assert stream_args.kwargs["headers"]["Range"] == (
            f"bytes=0-{_GGUF_HEADER_RANGE_BYTES - 1}"
        )
        assert stream_args.kwargs["headers"]["authorization"] == "Bearer test-token"
        response.raise_for_status.assert_called_once_with()

    def test_exact_selected_primary_header_pins_download_revision(self) -> None:
        from mobius.integrations.gguf._builder import (
            _preflight_hf_gguf_file,
            _resolve_gguf_path,
        )

        response = mock.MagicMock()
        response.iter_bytes.return_value = [_gguf_header_prefix("qwen35")]
        response_context = mock.MagicMock()
        response_context.__enter__.return_value = response
        session = mock.MagicMock()
        session.stream.return_value = response_context
        commit_hash = "a" * 40
        with (
            mock.patch(
                "mobius.integrations.gguf._builder.hf_hub_url",
                return_value="https://huggingface.co/exact-primary",
            ),
            mock.patch(
                "mobius.integrations.gguf._builder.get_hf_file_metadata",
                return_value=SimpleNamespace(
                    commit_hash=commit_hash,
                    location="https://huggingface.co/model.gguf",
                ),
            ),
            mock.patch(
                "mobius.integrations.gguf._builder.get_session",
                return_value=session,
            ),
        ):
            assert (
                _preflight_hf_gguf_file(
                    "example/qwen35",
                    "model.gguf",
                    revision="requested-revision",
                )
                == commit_hash
            )

        with (
            mock.patch(
                "mobius.integrations.gguf._builder._preflight_hf_gguf_file",
                return_value=commit_hash,
            ),
            mock.patch(
                "mobius.integrations.gguf._builder.hf_hub_download",
                return_value="cached-model.gguf",
            ) as download,
        ):
            assert _resolve_gguf_path("example/qwen35:model.gguf") == "cached-model.gguf"
        download.assert_called_once_with(
            repo_id="example/qwen35",
            filename="model.gguf",
            revision=commit_hash,
        )

    def test_explicit_hub_revision_and_filename_are_preflighted_exactly(self) -> None:
        from mobius.integrations.gguf._builder import _resolve_gguf_path

        commit_hash = "b" * 40
        with (
            mock.patch(
                "mobius.integrations.gguf._builder._preflight_hf_gguf_file",
                return_value=commit_hash,
            ) as preflight,
            mock.patch(
                "mobius.integrations.gguf._builder.hf_hub_download",
                return_value="cached-model.gguf",
            ) as download,
        ):
            assert (
                _resolve_gguf_path("example/qwen35@release-v1:nested/model.gguf")
                == "cached-model.gguf"
            )

        preflight.assert_called_once_with(
            "example/qwen35", "nested/model.gguf", revision="release-v1"
        )
        download.assert_called_once_with(
            repo_id="example/qwen35",
            filename="nested/model.gguf",
            revision=commit_hash,
        )

    def test_nested_hub_filename_is_preserved_for_runtime_identity(self):
        from mobius.integrations.gguf._builder import _logical_source_filename

        assert (
            _logical_source_filename(
                "example/model:nested/model.gguf",
                "/cache/blobs/abcdef",
            )
            == "nested/model.gguf"
        )
        assert (
            _logical_source_filename(
                "example/model",
                "/cache/models--example--model/snapshots/abc/nested/model.gguf",
            )
            == "nested/model.gguf"
        )

    def test_explicit_hub_revision_is_used_for_filename_discovery(self) -> None:
        from mobius.integrations.gguf._builder import _resolve_gguf_path

        commit_hash = "c" * 40
        with (
            mock.patch("mobius.integrations.gguf._builder.HfApi") as api_type,
            mock.patch(
                "mobius.integrations.gguf._builder._preflight_hf_gguf_file",
                return_value=commit_hash,
            ) as preflight,
            mock.patch(
                "mobius.integrations.gguf._builder.hf_hub_download",
                return_value="cached-historical.gguf",
            ) as download,
        ):
            api_type.return_value.list_repo_files.return_value = ["historical.gguf"]
            assert _resolve_gguf_path("example/qwen35@historical") == "cached-historical.gguf"

        api_type.return_value.list_repo_files.assert_called_once_with(
            "example/qwen35", revision="historical"
        )
        preflight.assert_called_once_with(
            "example/qwen35", "historical.gguf", revision="historical"
        )
        download.assert_called_once_with(
            repo_id="example/qwen35",
            filename="historical.gguf",
            revision=commit_hash,
        )

    def test_exact_selected_non_clip_header_rejects_without_payload(self) -> None:
        from mobius.integrations.gguf._builder import (
            _preflight_hf_mmproj_companion_file,
        )

        response = mock.MagicMock()
        response.iter_bytes.return_value = [_gguf_header_prefix("gemma4")]
        response_context = mock.MagicMock()
        response_context.__enter__.return_value = response
        session = mock.MagicMock()
        session.stream.return_value = response_context

        with (
            mock.patch(
                "mobius.integrations.gguf._builder.hf_hub_url",
                return_value="https://huggingface.co/selected-text-file",
            ),
            mock.patch(
                "mobius.integrations.gguf._builder.get_hf_file_metadata",
                return_value=SimpleNamespace(
                    commit_hash="d" * 40,
                    location="https://cdn.example/model.gguf",
                ),
            ),
            mock.patch(
                "mobius.integrations.gguf._builder.get_session",
                return_value=session,
            ),
            mock.patch(
                "mobius.integrations.gguf._builder.build_hf_headers",
                return_value={"authorization": "Bearer must-not-leak"},
            ),
            pytest.raises(
                ValueError,
                match=r"Expected a 'clip' mmproj.*architecture 'gemma4'.*No payload",
            ),
        ):
            _preflight_hf_mmproj_companion_file(
                "example/mixed-repo",
                "selected-text.gguf",
                revision="exact-revision",
            )

        assert "authorization" not in session.stream.call_args.kwargs["headers"]

    def test_structural_header_parser_rejects_duplicate_architecture_entries(self) -> None:
        from mobius.integrations.gguf._builder import (
            _gguf_architecture_from_header_prefix,
        )

        with pytest.raises(ValueError, match=r"exactly one.*found 2"):
            _gguf_architecture_from_header_prefix(
                _gguf_header_prefix("clip", "llama"),
                source="duplicate.gguf",
            )

    def test_structural_header_parser_rejects_array_above_safety_limit(self) -> None:
        from mobius.integrations.gguf._builder import (
            _gguf_architecture_from_header_prefix,
        )

        count = 1_000_001
        padding_entry = (
            struct.pack("<Q", len(b"padding"))
            + b"padding"
            + struct.pack("<IIQ", 9, 0, count)
            + bytes(count)
        )
        data = (
            b"GGUF"
            + struct.pack("<IQQ", 3, 0, 2)
            + padding_entry
            + _gguf_header_prefix("llama")[24:]
        )
        with pytest.raises(ValueError, match=r"1000001.*safety limit of 1000000"):
            _gguf_architecture_from_header_prefix(data, source="oversized-array.gguf")

    def test_structural_header_parser_rejects_truncated_fixed_width_array(self) -> None:
        from mobius.integrations.gguf._builder import (
            _gguf_architecture_from_header_prefix,
        )

        array_entry = (
            struct.pack("<Q", len(b"padding"))
            + b"padding"
            + struct.pack("<IIQ", 9, 4, 2)
            + b"\x00" * 7
        )
        data = b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + array_entry
        with pytest.raises(
            ValueError,
            match=r"truncated GGUF metadata array.*requiring at least 8 bytes.*7 remaining",
        ):
            _gguf_architecture_from_header_prefix(data, source="truncated-array.gguf")

    def test_structural_header_parser_rejects_truncated_nested_array(self) -> None:
        from mobius.integrations.gguf._builder import (
            _gguf_architecture_from_header_prefix,
        )

        nested_array = struct.pack("<IQ", 0, 2**63)
        array_entry = (
            struct.pack("<Q", len(b"padding"))
            + b"padding"
            + struct.pack("<IIQ", 9, 9, 1)
            + nested_array
        )
        data = b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + array_entry
        with pytest.raises(
            ValueError,
            match=r"truncated GGUF metadata array.*9223372036854775808 elements",
        ):
            _gguf_architecture_from_header_prefix(data, source="nested-array.gguf")

    def test_structural_header_parser_accepts_array_at_safety_limit(self) -> None:
        from mobius.integrations.gguf._builder import (
            _gguf_architecture_from_header_prefix,
        )

        count = 1_000_000
        padding_entry = (
            struct.pack("<Q", len(b"padding"))
            + b"padding"
            + struct.pack("<IIQ", 9, 0, count)
            + bytes(count)
        )
        data = (
            b"GGUF"
            + struct.pack("<IQQ", 3, 0, 2)
            + padding_entry
            + _gguf_header_prefix("llama")[24:]
        )
        assert (
            _gguf_architecture_from_header_prefix(data, source="boundary-array.gguf")
            == "llama"
        )

    def test_exact_companion_preflight_rejects_unresolved_mutable_revision(self) -> None:
        from mobius.integrations.gguf._builder import (
            _preflight_hf_mmproj_companion_file,
        )

        with (
            mock.patch(
                "mobius.integrations.gguf._builder.get_hf_file_metadata",
                side_effect=OfflineModeIsEnabled("offline"),
            ),
            pytest.raises(RuntimeError, match="immutable revision"),
        ):
            (
                _preflight_hf_mmproj_companion_file(
                    "example/offline",
                    "mmproj.gguf",
                    revision="main",
                )
            )

    def test_exact_companion_range_read_supports_requests_hub_sessions(self) -> None:
        from mobius.integrations.gguf._builder import (
            _preflight_hf_mmproj_companion_file,
        )

        class RequestsResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def raise_for_status(self):
                return None

            def iter_content(self, *, chunk_size: int):
                assert chunk_size == 64 * 1024
                yield _gguf_header_prefix("clip")

        session = mock.Mock()
        session.stream = False
        session.get.return_value = RequestsResponse()
        with (
            mock.patch(
                "mobius.integrations.gguf._builder.get_hf_file_metadata",
                return_value=SimpleNamespace(
                    commit_hash="e" * 40,
                    location="https://cdn.example/mmproj.gguf",
                ),
            ),
            mock.patch(
                "mobius.integrations.gguf._builder.get_session",
                return_value=session,
            ),
        ):
            assert (
                _preflight_hf_mmproj_companion_file(
                    "example/legacy-hub",
                    "mmproj.gguf",
                    revision="main",
                )
                == "e" * 40
            )

        session.get.assert_called_once()
        assert session.get.call_args.kwargs["stream"] is True

    def test_exact_companion_http_status_falls_back_to_pinned_download(self) -> None:
        from mobius.integrations.gguf._builder import (
            _preflight_hf_mmproj_companion_file,
        )

        request = httpx.Request("GET", "https://cdn.example/mmproj.gguf")
        failed_response = httpx.Response(403, request=request)
        response = mock.MagicMock()
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "forbidden",
            request=request,
            response=failed_response,
        )
        response_context = mock.MagicMock()
        response_context.__enter__.return_value = response
        session = mock.MagicMock()
        session.stream.return_value = response_context
        commit_hash = "f" * 40

        with (
            mock.patch(
                "mobius.integrations.gguf._builder.get_hf_file_metadata",
                return_value=SimpleNamespace(
                    commit_hash=commit_hash,
                    location="https://cdn.example/mmproj.gguf",
                ),
            ),
            mock.patch(
                "mobius.integrations.gguf._builder.get_session",
                return_value=session,
            ),
        ):
            assert (
                _preflight_hf_mmproj_companion_file(
                    "example/status-fallback",
                    "mmproj.gguf",
                    revision="main",
                )
                == commit_hash
            )

    def test_truncated_split_value_falls_back_to_pinned_download(self) -> None:
        from mobius.integrations.gguf._builder import _preflight_hf_gguf_file

        key = b"split.count"
        truncated = b"".join(
            [
                b"GGUF",
                struct.pack("<I", 3),
                struct.pack("<Q", 0),
                struct.pack("<Q", 1),
                struct.pack("<Q", len(key)),
                key,
                struct.pack("<I", 4),
                # UINT32 value deliberately omitted at the bounded range edge.
            ]
        )
        response = mock.MagicMock()
        response.iter_bytes.return_value = [truncated]
        response_context = mock.MagicMock()
        response_context.__enter__.return_value = response
        session = mock.MagicMock()
        session.stream.return_value = response_context
        commit_hash = "1" * 40

        with (
            mock.patch(
                "mobius.integrations.gguf._builder.get_hf_file_metadata",
                return_value=SimpleNamespace(
                    commit_hash=commit_hash,
                    location="https://cdn.example/model.gguf",
                ),
            ),
            mock.patch(
                "mobius.integrations.gguf._builder.get_session",
                return_value=session,
            ),
        ):
            assert (
                _preflight_hf_gguf_file(
                    "example/generic",
                    "model.gguf",
                    revision="main",
                )
                == commit_hash
            )

    @pytest.mark.parametrize(
        "preflight_error",
        [
            pytest.param(OfflineModeIsEnabled("offline"), id="offline-cache"),
            pytest.param(httpx.ConnectError("disconnected"), id="transport-error"),
        ],
    )
    def test_unavailable_exact_file_preflight_rejects_mutable_download(self, preflight_error):
        from mobius.integrations.gguf._builder import _resolve_gguf_path

        with (
            mock.patch(
                "mobius.integrations.gguf._builder.get_hf_file_metadata",
                side_effect=preflight_error,
            ),
            mock.patch(
                "mobius.integrations.gguf._builder.hf_hub_download",
                return_value="cached-model.gguf",
            ) as download,
            pytest.raises(RuntimeError, match="immutable revision"),
        ):
            _resolve_gguf_path("owner/repo:model.gguf")

        download.assert_not_called()

    def test_remote_incomplete_shard_fails_before_preflight_or_download(self):
        from mobius.integrations.gguf._builder import _resolve_gguf_path

        filename = "BF16/model-00001-of-00002.gguf"
        with (
            mock.patch("mobius.integrations.gguf._builder.HfApi") as api_type,
            mock.patch("mobius.integrations.gguf._builder.hf_hub_download") as download,
            mock.patch(
                "mobius.integrations.gguf._builder._preflight_hf_gguf_file"
            ) as preflight,
            pytest.raises(ValueError, match=r"Incomplete.*00002"),
        ):
            api_type.return_value.list_repo_files.return_value = [filename]
            _resolve_gguf_path(f"owner/repo:{filename}")

        preflight.assert_not_called()
        download.assert_not_called()

    def test_local_split_metadata_is_rejected(self):
        from mobius.integrations.gguf._builder import _raise_for_sharded_gguf

        with pytest.raises(NotImplementedError, match="cannot assemble split tensor tables"):
            _raise_for_sharded_gguf(source="model-00001-of-00002.gguf", split_count=2)


class TestHybridTensorContract:
    class _FakeGGUF:
        def __init__(self, architecture, metadata, tensor_names):
            self.architecture = architecture
            self.metadata = metadata
            self.tensor_names = tensor_names

    @staticmethod
    def _lfm2_names() -> list[str]:
        names = ["token_embd.weight", "token_embd_norm.weight"]
        common = [
            "attn_norm.weight",
            "ffn_norm.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ]
        mixers = [
            [
                "shortconv.conv.weight",
                "shortconv.in_proj.weight",
                "shortconv.out_proj.weight",
            ],
            [
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "attn_q_norm.weight",
                "attn_k_norm.weight",
            ],
        ]
        for layer, conditional in enumerate(mixers):
            names.extend(f"blk.{layer}.{suffix}" for suffix in [*common, *conditional])
        return names

    @classmethod
    def _lfm2moe_names(cls) -> list[str]:
        names = cls._lfm2_names()
        dense_layer_1 = {
            "blk.1.ffn_gate.weight",
            "blk.1.ffn_up.weight",
            "blk.1.ffn_down.weight",
        }
        names = [name for name in names if name not in dense_layer_1]
        names.extend(
            f"blk.1.{suffix}"
            for suffix in (
                "ffn_gate_inp.weight",
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
                "ffn_down_exps.weight",
                "exp_probs_b.bias",
            )
        )
        return names

    def test_lfm2_exact_mixer_closure_passes(self) -> None:
        from mobius.integrations.gguf._builder import (
            _raise_for_invalid_hybrid_tensor_contract,
        )

        model = self._FakeGGUF(
            "lfm2",
            {
                "lfm2.block_count": 2,
                "lfm2.attention.head_count_kv": [0, 2],
            },
            self._lfm2_names(),
        )
        _raise_for_invalid_hybrid_tensor_contract(model)

    def test_lfm2moe_exact_dense_and_routed_closure_passes(self) -> None:
        from mobius.integrations.gguf._builder import (
            _raise_for_invalid_hybrid_tensor_contract,
        )

        _raise_for_invalid_hybrid_tensor_contract(
            self._FakeGGUF(
                "lfm2moe",
                {
                    "lfm2moe.block_count": 2,
                    "lfm2moe.attention.head_count_kv": [0, 2],
                    "lfm2moe.leading_dense_block_count": 1,
                },
                self._lfm2moe_names(),
            )
        )

    @pytest.mark.parametrize(
        "wrong_tensor",
        ["blk.0.ffn_gate_inp.weight", "blk.1.ffn_gate.weight"],
    )
    def test_lfm2moe_wrong_ffn_family_is_rejected(self, wrong_tensor: str) -> None:
        from mobius.integrations.gguf._builder import (
            _raise_for_invalid_hybrid_tensor_contract,
        )

        with pytest.raises(ValueError, match=r"wrong .* FFN family"):
            _raise_for_invalid_hybrid_tensor_contract(
                self._FakeGGUF(
                    "lfm2moe",
                    {
                        "lfm2moe.block_count": 2,
                        "lfm2moe.attention.head_count_kv": [0, 2],
                        "lfm2moe.leading_dense_block_count": 1,
                    },
                    [*self._lfm2moe_names(), wrong_tensor],
                )
            )

    def test_wrong_mixer_tensor_is_rejected(self) -> None:
        from mobius.integrations.gguf._builder import (
            _raise_for_invalid_hybrid_tensor_contract,
        )

        names = self._lfm2_names()
        names.append("blk.0.attn_q.weight")
        model = self._FakeGGUF(
            "lfm2",
            {
                "lfm2.block_count": 2,
                "lfm2.attention.head_count_kv": [0, 2],
            },
            names,
        )
        with pytest.raises(ValueError, match="wrong conv mixer family"):
            _raise_for_invalid_hybrid_tensor_contract(model)

    def test_missing_recurrent_tensor_and_extra_layer_are_rejected(self) -> None:
        from mobius.integrations.gguf._builder import (
            _raise_for_invalid_hybrid_tensor_contract,
        )

        metadata = {
            "lfm2.block_count": 2,
            "lfm2.attention.head_count_kv": [0, 2],
        }
        missing = [
            name for name in self._lfm2_names() if name != "blk.0.shortconv.conv.weight"
        ]
        with pytest.raises(ValueError, match=r"shortconv\.conv\.weight"):
            _raise_for_invalid_hybrid_tensor_contract(
                self._FakeGGUF("lfm2", metadata, missing)
            )

        with pytest.raises(ValueError, match="out-of-range"):
            _raise_for_invalid_hybrid_tensor_contract(
                self._FakeGGUF(
                    "lfm2", metadata, [*self._lfm2_names(), "blk.2.ffn_norm.weight"]
                )
            )

    @staticmethod
    def _second_cohort_names(architecture: str) -> list[str]:
        names = ["token_embd.weight", "output_norm.weight"]
        if architecture == "jamba":
            common = [
                "attn_norm.weight",
                "ffn_norm.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            ]
            mixers = [
                [
                    "ssm_in.weight",
                    "ssm_conv1d.weight",
                    "ssm_conv1d.bias",
                    "ssm_x.weight",
                    "ssm_dt_norm.weight",
                    "ssm_dt.weight",
                    "ssm_dt.bias",
                    "ssm_b_norm.weight",
                    "ssm_c_norm.weight",
                    "ssm_a",
                    "ssm_d",
                    "ssm_out.weight",
                ],
                [
                    "attn_q.weight",
                    "attn_k.weight",
                    "attn_v.weight",
                    "attn_output.weight",
                ],
            ]
        elif architecture == "nemotron_h":
            common = ["attn_norm.weight"]
            mixers = [
                [
                    "ssm_in.weight",
                    "ssm_conv1d.weight",
                    "ssm_dt.bias",
                    "ssm_a",
                    "ssm_d",
                    "ssm_norm.weight",
                    "ssm_out.weight",
                ],
                ["ffn_up.weight", "ffn_down.weight"],
            ]
        else:
            common = [
                "attn_norm.weight",
                "ffn_norm.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            ]
            mixers = [
                [
                    "ssm_in.weight",
                    "ssm_conv1d.weight",
                    "ssm_dt.bias",
                    "ssm_a",
                    "ssm_d",
                    "ssm_norm.weight",
                    "ssm_out.weight",
                ],
                [
                    "attn_q.weight",
                    "attn_q.bias",
                    "attn_k.weight",
                    "attn_k.bias",
                    "attn_v.weight",
                    "attn_v.bias",
                    "attn_output.weight",
                ],
            ]
        for layer, mixer in enumerate(mixers):
            names.extend(f"blk.{layer}.{suffix}" for suffix in [*common, *mixer])
        return names

    @pytest.mark.parametrize("architecture", ["jamba", "nemotron_h", "granitehybrid"])
    def test_second_cohort_exact_closure_and_mutations(self, architecture: str) -> None:
        from mobius.integrations.gguf._builder import (
            _raise_for_invalid_hybrid_tensor_contract,
        )

        metadata = {
            f"{architecture}.block_count": 2,
            f"{architecture}.attention.head_count_kv": [0, 2],
        }
        if architecture == "nemotron_h":
            metadata[f"{architecture}.feed_forward_length"] = [0, 128]
        names = self._second_cohort_names(architecture)
        _raise_for_invalid_hybrid_tensor_contract(
            self._FakeGGUF(architecture, metadata, names)
        )

        with pytest.raises(ValueError, match="missing"):
            _raise_for_invalid_hybrid_tensor_contract(
                self._FakeGGUF(architecture, metadata, names[:-1])
            )
        with pytest.raises(ValueError, match="unexpected"):
            _raise_for_invalid_hybrid_tensor_contract(
                self._FakeGGUF(
                    architecture,
                    metadata,
                    [*names, "blk.0.attn_q.weight"],
                )
            )
        mtp_metadata = {
            **metadata,
            f"{architecture}.nextn_predict_layers": 1,
        }
        with pytest.raises(ValueError, match=r"auxiliary|folded MTP"):
            _raise_for_invalid_hybrid_tensor_contract(
                self._FakeGGUF(architecture, mtp_metadata, names)
            )
        with pytest.raises(ValueError, match="out_of_range"):
            _raise_for_invalid_hybrid_tensor_contract(
                self._FakeGGUF(
                    architecture,
                    metadata,
                    [*names, "blk.2.attn_norm.weight"],
                )
            )
        partial_bias = {
            "nemotron_h": "blk.1.ffn_up.bias",
        }.get(architecture)
        if partial_bias is not None:
            with pytest.raises(ValueError, match=r"partial .* bias family"):
                _raise_for_invalid_hybrid_tensor_contract(
                    self._FakeGGUF(
                        architecture,
                        metadata,
                        [*names, partial_bias],
                    )
                )

        if architecture == "granitehybrid":
            dense_biases = [
                f"blk.{layer}.ffn_{projection}.bias"
                for layer in range(2)
                for projection in ("gate", "up", "down")
            ]
            _raise_for_invalid_hybrid_tensor_contract(
                self._FakeGGUF(architecture, metadata, [*names, *dense_biases])
            )
            with pytest.raises(ValueError, match=r"partial dense shared-MLP bias family"):
                _raise_for_invalid_hybrid_tensor_contract(
                    self._FakeGGUF(
                        architecture,
                        metadata,
                        [*names, *dense_biases[:-1]],
                    )
                )

    def test_partial_fused_and_separate_experts_are_rejected(self) -> None:
        from mobius.integrations.gguf._builder import (
            _raise_for_invalid_hybrid_tensor_contract,
        )

        model = self._FakeGGUF(
            "qwen35moe",
            {
                "qwen35moe.block_count": 1,
                "qwen35moe.attention.recurrent_layers": [False],
            },
            [
                "token_embd.weight",
                "output_norm.weight",
                "blk.0.ffn_gate_up_exps.weight",
                "blk.0.ffn_gate_exps.weight",
            ],
        )
        with pytest.raises(ValueError, match="mixes fused and separate"):
            _raise_for_invalid_hybrid_tensor_contract(model)

    def test_partial_modern_and_legacy_recurrent_inputs_are_rejected(self) -> None:
        from mobius.integrations.gguf._builder import (
            _raise_for_invalid_hybrid_tensor_contract,
        )

        model = self._FakeGGUF(
            "qwen3next",
            {
                "qwen3next.block_count": 1,
                "qwen3next.attention.recurrent_layers": [True],
            },
            [
                "token_embd.weight",
                "output_norm.weight",
                "blk.0.ffn_gate_up_exps.weight",
                "blk.0.ssm_in.weight",
                "blk.0.attn_qkv.weight",
            ],
        )
        with pytest.raises(ValueError, match="mixes legacy ssm_in with modern"):
            _raise_for_invalid_hybrid_tensor_contract(model)


class TestNormalizeGgufWeights:
    """Tests for GGUF-specific weight shape/value normalization."""

    def test_deltanet_a_log_is_inverted_from_neg_exp(self):
        """GGUF ssm_a = -exp(A_log); normalize must recover raw A_log = log(-ssm_a).

        The GatedDeltaNet module re-applies ``-exp(A_log)`` at runtime, so the
        round-trip ``-exp(normalize(ssm_a))`` must reproduce the original
        ``ssm_a`` the converter stored.
        """
        import torch

        from mobius.integrations.gguf._builder import _normalize_gguf_weights

        # A representative raw A_log, and the value the GGUF converter stores.
        a_log_raw = torch.tensor([-3.4688, -1.0703, -5.0, -0.5], dtype=torch.float32)
        ssm_a = -torch.exp(a_log_raw)  # what llama.cpp writes to blk.N.ssm_a
        assert bool((ssm_a < 0).all())  # sanity: pre-transformed value is negative

        key = "model.layers.0.linear_attn.A_log"
        out = _normalize_gguf_weights({key: ssm_a})

        # The stored parameter must be the raw A_log again ...
        assert torch.allclose(out[key], a_log_raw, atol=1e-5)
        # ... so that the module's runtime -exp(A_log) recovers ssm_a exactly.
        assert torch.allclose(-torch.exp(out[key]), ssm_a, atol=1e-6)

    def test_non_deltanet_a_log_is_untouched(self):
        """Mamba/PLaMo SSM ``A_log`` (consumed as ``A`` directly) must not be inverted."""
        import torch

        from mobius.integrations.gguf._builder import _normalize_gguf_weights

        ssm_a = torch.tensor([-0.04, -0.5], dtype=torch.float32)
        key = "backbone.layers.0.mixer.A_log"
        out = _normalize_gguf_weights({key: ssm_a})
        assert torch.allclose(out[key], ssm_a)

    def test_zero_centered_norm_offset_removed_for_qwen35(self):
        """qwen35 GGUF bakes +1 into transformer norms; normalize must strip it.

        mobius applies the ``1 +`` at runtime via OffsetRMSNorm, so the stored
        weight must be the raw zero-centered value again.
        """
        import torch

        from mobius.integrations.gguf._builder import _normalize_gguf_weights

        sd = {
            "model.layers.0.input_layernorm.weight": torch.tensor([1.5, 2.0]),
            "model.layers.3.self_attn.q_norm.weight": torch.tensor([1.25]),
            "model.layers.3.self_attn.k_norm.weight": torch.tensor([1.1]),
            "model.norm.weight": torch.tensor([1.94]),
            # DeltaNet internal gated norm — converter did NOT add +1.
            "model.layers.0.linear_attn.norm.weight": torch.tensor([0.87]),
        }
        out = _normalize_gguf_weights(dict(sd), gguf_arch="qwen35")

        assert torch.allclose(
            out["model.layers.0.input_layernorm.weight"], torch.tensor([0.5, 1.0])
        )
        assert torch.allclose(
            out["model.layers.3.self_attn.q_norm.weight"], torch.tensor([0.25])
        )
        assert torch.allclose(out["model.norm.weight"], torch.tensor([0.94]))
        # linear_attn.norm is a plain gated RMSNorm — must be left untouched.
        assert torch.allclose(
            out["model.layers.0.linear_attn.norm.weight"], torch.tensor([0.87])
        )

    def test_norm_offset_not_applied_for_non_offset_arch(self):
        """Standard-RMSNorm archs (e.g. llama/qwen2) must not have norms shifted."""
        import torch

        from mobius.integrations.gguf._builder import _normalize_gguf_weights

        sd = {
            "model.layers.0.input_layernorm.weight": torch.tensor([1.0, 1.0]),
            "model.norm.weight": torch.tensor([1.0]),
        }
        out = _normalize_gguf_weights(dict(sd), gguf_arch="qwen2")
        assert torch.allclose(
            out["model.layers.0.input_layernorm.weight"], torch.tensor([1.0, 1.0])
        )
        assert torch.allclose(out["model.norm.weight"], torch.tensor([1.0]))

    @pytest.mark.parametrize(
        ("suffix", "shape"),
        [
            ("weight", (2, 6, 4)),
            ("qweight", (2, 6, 4, 2)),
            ("scales", (2, 6, 4)),
            ("zero_points", (2, 6, 4)),
        ],
    )
    def test_fused_experts_are_split_without_tensor_loss(self, suffix, shape):
        import torch

        from mobius.integrations.gguf._builder import _normalize_gguf_weights

        key = f"model.layers.0.mlp.experts.gate_up_proj.{suffix}"
        value = torch.arange(np.prod(shape)).reshape(shape)
        out = _normalize_gguf_weights({key: value}, gguf_arch="qwen35moe")

        assert key not in out
        assert len(out) == 4
        gate, up = value.chunk(2, dim=1)
        for expert in range(2):
            assert torch.equal(
                out[f"model.layers.0.mlp.experts.{expert}.gate_proj.{suffix}"],
                gate[expert],
            )
            assert torch.equal(
                out[f"model.layers.0.mlp.experts.{expert}.up_proj.{suffix}"],
                up[expert],
            )


class TestReorderDeltaNetVHeads:
    """Undo of the GGUF converter's grouped→tiled Gated-DeltaNet V-head order.

    The llama.cpp converter reorders every V-indexed ``linear_attn`` tensor from
    HuggingFace *grouped* order into ggml *tiled* order (see
    ``_LinearAttentionVReorderBase._reorder_v_heads``). mobius consumes grouped
    order, so ``_reorder_deltanet_v_heads`` must be the exact inverse.
    """

    # Small grouped linear-attention geometry: 2 K-heads, 6 V-heads (v_per_k=3).
    CFG = SimpleNamespace(
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
    )

    @staticmethod
    def _converter_reorder(tensor, dim, num_k_heads, num_v_per_k, head_dim):
        """Reference grouped→tiled reorder copied from llama.cpp's converter."""
        import torch  # noqa: F401

        shape = list(tensor.shape)
        if dim < 0:
            dim += len(shape)
        new_shape = [*shape[:dim], num_k_heads, num_v_per_k, head_dim, *shape[dim + 1 :]]
        tensor = tensor.reshape(*new_shape)
        perm = list(range(len(new_shape)))
        perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
        return tensor.permute(*perm).contiguous().reshape(*shape)

    def test_row_tensors_roundtrip(self):
        """Grouped weights survive tile→untile for every V-row projection."""
        import torch

        from mobius.integrations.gguf._builder import _reorder_deltanet_v_heads

        cfg = self.CFG
        n_k, n_v = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        v_per_k = n_v // n_k
        hd_k, hd_v = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        key_dim, value_dim = hd_k * n_k, hd_v * n_v
        hidden = 5
        torch.manual_seed(0)

        p = "model.layers.0.linear_attn."
        grouped = {
            f"{p}in_proj_z.weight": torch.randn(value_dim, hidden),
            f"{p}in_proj_a.weight": torch.randn(n_v, hidden),
            f"{p}in_proj_b.weight": torch.randn(n_v, hidden),
            f"{p}A_log": torch.randn(n_v),
            f"{p}dt_bias": torch.randn(n_v),
            f"{p}conv1d.weight": torch.randn(2 * key_dim + value_dim, 1, 4),
        }
        # in_proj_qkv: only the V rows (after 2*key_dim) are reordered.
        qkv = torch.randn(2 * key_dim + value_dim, hidden)
        grouped[f"{p}in_proj_qkv.weight"] = qkv

        # Build the tiled (GGUF) state by applying the converter's reorder.
        tiled = {k: v.clone() for k, v in grouped.items()}
        tiled[f"{p}in_proj_z.weight"] = self._converter_reorder(
            grouped[f"{p}in_proj_z.weight"], 0, n_k, v_per_k, hd_v
        )
        for name in ("in_proj_a", "in_proj_b"):
            tiled[f"{p}{name}.weight"] = self._converter_reorder(
                grouped[f"{p}{name}.weight"], 0, n_k, v_per_k, 1
            )
        for name in ("A_log", "dt_bias"):
            tiled[f"{p}{name}"] = self._converter_reorder(
                grouped[f"{p}{name}"], 0, n_k, v_per_k, 1
            )
        # V portion of qkv / conv1d.
        v0 = 2 * key_dim
        qv = self._converter_reorder(qkv[v0:], 0, n_k, v_per_k, hd_v)
        tiled[f"{p}in_proj_qkv.weight"] = torch.cat([qkv[:v0], qv], dim=0)
        conv = grouped[f"{p}conv1d.weight"]
        cv = self._converter_reorder(conv[v0:], 0, n_k, v_per_k, hd_v)
        tiled[f"{p}conv1d.weight"] = torch.cat([conv[:v0], cv], dim=0)

        out = _reorder_deltanet_v_heads({k: v.clone() for k, v in tiled.items()}, cfg)

        for k in grouped:
            assert torch.allclose(out[k], grouped[k]), k

    def test_quantized_out_proj_columns_roundtrip(self):
        """out_proj's quantized K axis (blocks + packed zero-points) round-trips."""
        import torch

        from mobius.integrations.gguf._builder import _reorder_deltanet_v_heads

        cfg = self.CFG
        n_k, n_v = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        v_per_k = n_v // n_k
        hd_v = cfg.linear_value_head_dim  # 4
        value_dim = hd_v * n_v  # 24
        block = 2  # 2 elems/block -> head_v_dim(4) = 2 blocks (even -> byte aligned)
        n_blocks = value_dim // block  # 12
        hidden = 5
        torch.manual_seed(1)

        p = "model.layers.0.linear_attn."
        # Grouped quantized out_proj triplet: [hidden, K/block, block/2], etc.
        gw = torch.randint(0, 255, (hidden, n_blocks, block // 2 + 7), dtype=torch.uint8)
        gs = torch.randn(hidden, n_blocks, dtype=torch.float16)
        gz = torch.randint(0, 255, (hidden, n_blocks // 2), dtype=torch.uint8)
        # Provide a grouped row tensor so the head geometry is exercised too.
        grouped = {
            f"{p}out_proj.weight": gw,
            f"{p}out_proj.scales": gs,
            f"{p}out_proj.zero_points": gz,
        }
        blocks_per_head = n_blocks // n_v  # 2
        tiled = {
            f"{p}out_proj.weight": self._converter_reorder(
                gw, 1, n_k, v_per_k, blocks_per_head
            ),
            f"{p}out_proj.scales": self._converter_reorder(
                gs, 1, n_k, v_per_k, blocks_per_head
            ),
            f"{p}out_proj.zero_points": self._converter_reorder(
                gz, 1, n_k, v_per_k, blocks_per_head // 2
            ),
        }

        out = _reorder_deltanet_v_heads({k: v.clone() for k, v in tiled.items()}, cfg)

        for k in grouped:
            assert torch.equal(out[k], grouped[k]), k

    def test_no_reorder_when_heads_equal(self):
        """Ungrouped linear attention (num_v == num_k) is left untouched."""
        import torch

        from mobius.integrations.gguf._builder import _reorder_deltanet_v_heads

        cfg = SimpleNamespace(
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
        )
        p = "model.layers.0.linear_attn."
        sd = {f"{p}in_proj_z.weight": torch.randn(16, 5)}
        ref = sd[f"{p}in_proj_z.weight"].clone()
        out = _reorder_deltanet_v_heads({k: v.clone() for k, v in sd.items()}, cfg)
        assert torch.equal(out[f"{p}in_proj_z.weight"], ref)


class TestGgufArchSurvivesToWeightProcessing:
    """``_gguf_arch`` must reach ``process_tensors`` on every build path.

    It is a plain instance attribute, not a dataclass field, so every
    ``dataclasses.replace`` in the builder drops it. It is also the key the
    weight-processor dispatch is built on, so losing it silently demotes
    dispatch to the ``model_type`` fallback — which is exactly the indirection
    the architecture registry exists to remove. A regression here would be
    invisible until a spec's processor stopped agreeing with its model_type,
    and would then affect only non-float32 and quantized imports.
    """

    @staticmethod
    def _recorded_arches(monkeypatch, gguf_path: Path, **build_kwargs) -> list[object]:
        from mobius.integrations.gguf import _builder as builder_module
        from mobius.integrations.gguf import _tensor_processors

        seen: list[object] = []
        real = _tensor_processors.process_tensors

        def spy(state_dict, config):
            seen.append(getattr(config, "_gguf_arch", None))
            return real(state_dict, config)

        monkeypatch.setattr(_tensor_processors, "process_tensors", spy)
        builder_module.build_from_gguf(gguf_path, **build_kwargs)
        return seen

    def test_float_path_keeps_the_architecture(self, monkeypatch, q4_0_gguf: Path):
        seen = self._recorded_arches(monkeypatch, q4_0_gguf, keep_quantized=False)
        assert seen, "process_tensors was never called"
        assert all(arch == "llama" for arch in seen), seen

    def test_dtype_override_keeps_the_architecture(self, monkeypatch, q4_0_gguf: Path):
        """``dtype`` triggers a ``dataclasses.replace`` that drops the attribute."""
        seen = self._recorded_arches(
            monkeypatch, q4_0_gguf, keep_quantized=False, dtype="float16"
        )
        assert seen, "process_tensors was never called"
        assert all(arch == "llama" for arch in seen), seen

    def test_quantized_path_keeps_the_architecture(self, monkeypatch, q4_0_gguf: Path):
        """The preserve-quantization path replaces the config as well."""
        seen = self._recorded_arches(monkeypatch, q4_0_gguf, keep_quantized=True)
        assert seen, "process_tensors was never called"
        assert all(arch == "llama" for arch in seen), seen
