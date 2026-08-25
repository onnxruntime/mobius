# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for GGUF ``clip`` mmproj config extraction and vision-encoder build.

Builds a small synthetic ``clip`` mmproj GGUF with :class:`GGUFWriter` (mirroring
``_builder_test.py``), then exercises the mmproj config readers and the vision
encoder build+run path end-to-end on CPU.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

# Small synthetic vision encoder dimensions.
_VISION_HIDDEN = 16
_VISION_FFN = 32
_VISION_LAYERS = 2
_VISION_HEADS = 2
_IMAGE_SIZE = 16
_PATCH_SIZE = 4
_POS_EMB_SIZE = 8
_TEXT_HIDDEN = 32

# Small synthetic audio encoder dimensions.
_AUDIO_HIDDEN = 16
_AUDIO_FFN = 32
_AUDIO_LAYERS = 2
_AUDIO_HEADS = 2
_NUM_MEL_BINS = 8
_AUDIO_CONV0 = 16
_AUDIO_CONV1 = 16
_AUDIO_PROJ_OUT = _TEXT_HIDDEN


def _write_clip_mmproj_gguf(
    path: Path,
    *,
    with_audio: bool = True,
    vision_projector_type: str = "gemma4v",
    extra_tensor: str | None = None,
    identity_name: str = "Gemma-4-E2B-It",
    identity_repo: str | None = "https://huggingface.co/google/gemma-4-E2B-it",
    patch_weight_shape: tuple[int, ...] | None = None,
    audio_num_mel_bins: int | None = _NUM_MEL_BINS,
) -> None:
    """Write a small synthetic Gemma4 ``clip`` mmproj GGUF for tests."""
    from gguf import GGUFWriter

    head_dim = _VISION_HIDDEN // _VISION_HEADS
    writer = GGUFWriter(str(path), "clip")
    writer.add_string("general.name", identity_name)
    writer.add_string("general.base_model.0.name", identity_name)
    if identity_repo is not None:
        writer.add_string("general.base_model.0.repo_url", identity_repo)
    writer.add_string("general.type", "mmproj")

    # --- vision metadata ---
    writer.add_bool("clip.has_vision_encoder", True)
    writer.add_string("clip.vision.projector_type", vision_projector_type)
    writer.add_uint32("clip.vision.embedding_length", _VISION_HIDDEN)
    writer.add_uint32("clip.vision.feed_forward_length", _VISION_FFN)
    writer.add_uint32("clip.vision.block_count", _VISION_LAYERS)
    writer.add_uint32("clip.vision.attention.head_count", _VISION_HEADS)
    writer.add_uint32("clip.vision.image_size", _IMAGE_SIZE)
    writer.add_uint32("clip.vision.patch_size", _PATCH_SIZE)
    writer.add_uint32("clip.vision.projection_dim", _TEXT_HIDDEN)
    writer.add_array("clip.vision.image_mean", [0.0, 0.0, 0.0])
    writer.add_array("clip.vision.image_std", [1.0, 1.0, 1.0])
    writer.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-6)

    def _f32(name: str, shape: tuple[int, ...]) -> None:
        writer.add_tensor(name, np.random.randn(*shape).astype(np.float32))

    # patch embed (conv layout) + position table + projector.
    _f32(
        "v.patch_embd.weight",
        patch_weight_shape or (_VISION_HIDDEN, 3, _PATCH_SIZE, _PATCH_SIZE),
    )
    _f32("v.position_embd.weight", (2, _POS_EMB_SIZE, _VISION_HIDDEN))
    _f32("mm.input_projection.weight", (_TEXT_HIDDEN, _VISION_HIDDEN))
    for layer in range(_VISION_LAYERS):
        prefix = f"v.blk.{layer}."
        for norm in ("ln1", "ln2", "attn_post_norm", "ffn_post_norm"):
            _f32(prefix + norm + ".weight", (_VISION_HIDDEN,))
        for proj in ("attn_q", "attn_k", "attn_v", "attn_out"):
            _f32(prefix + proj + ".weight", (_VISION_HIDDEN, _VISION_HIDDEN))
            for bound in ("input_min", "input_max", "output_min", "output_max"):
                _f32(prefix + proj + "." + bound, (1,))
        for qk_norm in ("attn_q_norm", "attn_k_norm"):
            _f32(prefix + qk_norm + ".weight", (head_dim,))
        _f32(prefix + "ffn_gate.weight", (_VISION_FFN, _VISION_HIDDEN))
        _f32(prefix + "ffn_up.weight", (_VISION_FFN, _VISION_HIDDEN))
        _f32(prefix + "ffn_down.weight", (_VISION_HIDDEN, _VISION_FFN))
        for proj in ("ffn_gate", "ffn_up", "ffn_down"):
            for bound in ("input_min", "input_max", "output_min", "output_max"):
                _f32(prefix + proj + "." + bound, (1,))

    # --- audio metadata (best-effort, for config-extraction tests) ---
    if with_audio:
        writer.add_bool("clip.has_audio_encoder", True)
        writer.add_string("clip.audio.projector_type", "gemma4a")
        writer.add_uint32("clip.audio.embedding_length", _AUDIO_HIDDEN)
        writer.add_uint32("clip.audio.feed_forward_length", _AUDIO_FFN)
        writer.add_uint32("clip.audio.block_count", _AUDIO_LAYERS)
        writer.add_uint32("clip.audio.attention.head_count", _AUDIO_HEADS)
        if audio_num_mel_bins is not None:
            writer.add_uint32("clip.audio.num_mel_bins", audio_num_mel_bins)
        writer.add_uint32("clip.audio.projection_dim", _TEXT_HIDDEN)
        writer.add_float32("clip.audio.attention.layer_norm_epsilon", 1e-6)
        _f32("a.conv1d.0.weight", (_AUDIO_CONV0, 1, 3, 3))
        _f32("a.conv1d.0.norm.weight", (_AUDIO_CONV0,))
        _f32("a.conv1d.1.weight", (_AUDIO_CONV1, _AUDIO_CONV0, 3, 3))
        _f32("a.conv1d.1.norm.weight", (_AUDIO_CONV1,))
        _f32("a.input_projection.weight", (_AUDIO_HIDDEN, _AUDIO_HIDDEN))
        _f32("a.pre_encode.out.weight", (_AUDIO_PROJ_OUT, _AUDIO_HIDDEN))
        _f32("a.pre_encode.out.bias", (_AUDIO_PROJ_OUT,))
        _f32("mm.a.input_projection.weight", (_TEXT_HIDDEN, _AUDIO_PROJ_OUT))
        audio_head_dim = _AUDIO_HIDDEN // _AUDIO_HEADS
        clipped_stems = (
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_out",
            "conv_pw1",
            "conv_pw2",
            "ffn_up",
            "ffn_down",
            "ffn_up_1",
            "ffn_down_1",
        )
        for layer in range(_AUDIO_LAYERS):
            prefix = f"a.blk.{layer}."
            for norm in (
                "ffn_norm",
                "ffn_post_norm",
                "ffn_norm_1",
                "ffn_post_norm_1",
                "attn_pre_norm",
                "attn_post_norm",
                "ln2",
                "norm_conv",
            ):
                _f32(prefix + norm + ".weight", (_AUDIO_HIDDEN,))
            _f32(prefix + "per_dim_scale.weight", (audio_head_dim,))
            for stem in ("attn_q", "attn_k", "attn_v", "attn_out", "attn_k_rel"):
                _f32(prefix + stem + ".weight", (_AUDIO_HIDDEN, _AUDIO_HIDDEN))
            _f32(prefix + "conv_pw1.weight", (2 * _AUDIO_HIDDEN, _AUDIO_HIDDEN))
            _f32(prefix + "conv_dw.weight", (_AUDIO_HIDDEN, 5))
            _f32(prefix + "conv_pw2.weight", (_AUDIO_HIDDEN, _AUDIO_HIDDEN))
            for stem in ("ffn_up", "ffn_up_1"):
                _f32(prefix + stem + ".weight", (_AUDIO_FFN, _AUDIO_HIDDEN))
            for stem in ("ffn_down", "ffn_down_1"):
                _f32(prefix + stem + ".weight", (_AUDIO_HIDDEN, _AUDIO_FFN))
            for stem in clipped_stems:
                for bound in ("input_min", "input_max", "output_min", "output_max"):
                    _f32(prefix + stem + "." + bound, (1,))
    if extra_tensor is not None:
        _f32(extra_tensor, (1,))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _write_minimal_gguf(
    path: Path,
    architecture: str,
    *,
    split_count: int = 1,
    identity_name: str | None = None,
    identity_repo: str | None = None,
) -> None:
    """Write only enough GGUF structure to exercise pre-config guards."""
    from gguf import GGUFWriter

    writer = GGUFWriter(str(path), architecture)
    if identity_name is None:
        identity_name = {
            "gemma4": "Gemma-4-E2B-It",
            "muse-glimmer": "Muse-Glimmer-30B",
            "muse_glimmer": "Muse-Glimmer-30B",
        }.get(architecture)
    if identity_repo is None and architecture == "gemma4":
        identity_repo = "https://huggingface.co/google/gemma-4-E2B-it"
    if identity_name is not None:
        writer.add_string("general.name", identity_name)
        if architecture == "gemma4" or identity_repo is not None:
            writer.add_string("general.base_model.0.name", identity_name)
    if identity_repo is not None:
        writer.add_string("general.base_model.0.repo_url", identity_repo)
    if split_count > 1:
        writer.add_uint16("split.no", 0)
        writer.add_uint16("split.count", split_count)
        writer.add_uint64("split.tensors.count", 1)
    writer.add_tensor("sentinel", np.zeros((1,), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _write_quantized_gemma4_text_gguf(path: Path) -> None:
    """Write a tiny Gemma4 text GGUF with Q4 projections and a float embedding."""
    from gguf import GGMLQuantizationType, GGUFWriter

    hidden_size = 32
    intermediate_size = 64
    vocab_size = 64
    num_layers = 2
    num_heads = 4
    num_kv_heads = 1
    head_dim = hidden_size // num_heads

    writer = GGUFWriter(str(path), "gemma4")
    writer.add_string("general.name", "Gemma-4-E2B-It")
    writer.add_string("general.base_model.0.name", "Gemma-4-E2B-It")
    writer.add_string(
        "general.base_model.0.repo_url",
        "https://huggingface.co/google/gemma-4-E2B-it",
    )
    writer.add_context_length(128)
    writer.add_embedding_length(hidden_size)
    writer.add_feed_forward_length(intermediate_size)
    writer.add_block_count(num_layers)
    writer.add_head_count(num_heads)
    writer.add_head_count_kv(num_kv_heads)
    writer.add_key_length(head_dim)
    writer.add_key_length_swa(head_dim)
    writer.add_rope_dimension_count(head_dim)
    writer.add_rope_dimension_count_swa(head_dim)
    writer.add_rope_freq_base(10_000.0)
    writer.add_rope_freq_base_swa(10_000.0)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_vocab_size(vocab_size)
    writer.add_array(
        "gemma4.attention.sliding_window_pattern",
        [True] * num_layers,
    )

    def _f32(name: str, shape: tuple[int, ...]) -> None:
        writer.add_tensor(name, np.random.randn(*shape).astype(np.float32))

    def _q4_0(name: str, n_out: int, k_in: int) -> None:
        block_size = 32
        block_bytes = 18
        raw = np.zeros((n_out, k_in // block_size * block_bytes), dtype=np.uint8)
        for row in range(n_out):
            for block in range(k_in // block_size):
                offset = block * block_bytes
                raw[row, offset : offset + 2] = np.array(
                    [np.random.uniform(0.01, 1.0)],
                    dtype=np.float16,
                ).view(np.uint8)
                raw[row, offset + 2 : offset + block_bytes] = np.random.randint(
                    0,
                    256,
                    size=block_bytes - 2,
                    dtype=np.uint8,
                )
        writer.add_tensor(name, raw, raw_dtype=GGMLQuantizationType.Q4_0)

    # The float embedding is deliberately incompatible with the projection
    # Q4_0 target, so the graph must retain a normal float embedding initializer.
    _f32("token_embd.weight", (vocab_size, hidden_size))
    _q4_0("output.weight", vocab_size, hidden_size)
    _f32("output_norm.weight", (hidden_size,))

    for layer in range(num_layers):
        prefix = f"blk.{layer}"
        _q4_0(f"{prefix}.attn_q.weight", num_heads * head_dim, hidden_size)
        _q4_0(f"{prefix}.attn_k.weight", num_kv_heads * head_dim, hidden_size)
        _q4_0(f"{prefix}.attn_v.weight", num_kv_heads * head_dim, hidden_size)
        _q4_0(f"{prefix}.attn_output.weight", hidden_size, num_heads * head_dim)
        _q4_0(f"{prefix}.ffn_gate.weight", intermediate_size, hidden_size)
        _q4_0(f"{prefix}.ffn_up.weight", intermediate_size, hidden_size)
        _q4_0(f"{prefix}.ffn_down.weight", hidden_size, intermediate_size)
        for norm in (
            "attn_norm",
            "post_attention_norm",
            "ffn_norm",
            "post_ffw_norm",
        ):
            _f32(f"{prefix}.{norm}.weight", (hidden_size,))
        for norm in ("attn_q_norm", "attn_k_norm"):
            _f32(f"{prefix}.{norm}.weight", (head_dim,))
        _f32(f"{prefix}.layer_output_scale.weight", (1,))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.fixture
def clip_mmproj_gguf(tmp_path: Path) -> Path:
    path = tmp_path / "mmproj.gguf"
    _write_clip_mmproj_gguf(path)
    return path


class TestMultimodalPreflightGuards:
    @staticmethod
    def _load_pair(text_path: Path, mmproj_path: Path):
        from mobius.integrations.gguf._reader import GGUFModel

        return GGUFModel(str(text_path)), GGUFModel(str(mmproj_path))

    def test_clip_is_allowed_only_in_mmproj_companion_context(
        self, clip_mmproj_gguf: Path
    ) -> None:
        from mobius.integrations.gguf._builder import _validate_gguf_model
        from mobius.integrations.gguf._errors import DisabledGGUFArchitectureError
        from mobius.integrations.gguf._reader import GGUFModel

        mmproj = GGUFModel(str(clip_mmproj_gguf))
        with pytest.raises(DisabledGGUFArchitectureError, match="intentionally disabled"):
            _validate_gguf_model(mmproj, source=str(clip_mmproj_gguf))

        _validate_gguf_model(
            mmproj,
            source=str(clip_mmproj_gguf),
            allow_mmproj_companion=True,
        )

    def test_rejects_non_gemma4_text_architecture_before_config(
        self,
        tmp_path: Path,
    ):
        from mobius.integrations.gguf._mmproj import build_gemma4_vlm_from_gguf

        text_path = tmp_path / "nemotron-h-moe.gguf"
        _write_minimal_gguf(text_path, "nemotron_h_moe")

        with (
            mock.patch(
                "mobius.integrations.gguf._mmproj._resolve_local_path",
                side_effect=[str(text_path)],
            ) as resolve,
            pytest.raises(ValueError, match="requires a gemma4 text GGUF"),
        ):
            build_gemma4_vlm_from_gguf(text_path, "owner/repo:mmproj.gguf")

        assert resolve.call_count == 1

    def test_rejects_split_mmproj_before_config(self, tmp_path: Path):
        from mobius.integrations.gguf._mmproj import build_gemma4_vlm_from_gguf

        text_path = tmp_path / "gemma4.gguf"
        mmproj_path = tmp_path / "mmproj-00001-of-00002.gguf"
        _write_minimal_gguf(text_path, "gemma4")
        _write_minimal_gguf(mmproj_path, "clip", split_count=2)

        with pytest.raises(NotImplementedError, match="cannot assemble split tensor tables"):
            build_gemma4_vlm_from_gguf(text_path, mmproj_path)

    @pytest.mark.parametrize("projector_type", ["unknown-projector", "mlp", "gemma4a"])
    def test_unknown_or_deferred_projector_fails_before_graph_construction(
        self, tmp_path: Path, projector_type: str
    ):
        from mobius.integrations.gguf._mmproj import build_gemma4_vlm_from_gguf

        text_path = tmp_path / "gemma4.gguf"
        mmproj_path = tmp_path / "mmproj.gguf"
        _write_minimal_gguf(text_path, "gemma4")
        _write_clip_mmproj_gguf(mmproj_path, vision_projector_type=projector_type)

        with (
            mock.patch("mobius._builder.build_from_module") as build_graph,
            pytest.raises((ValueError, NotImplementedError), match=projector_type),
        ):
            build_gemma4_vlm_from_gguf(text_path, mmproj_path, image_token_id=0)
        build_graph.assert_not_called()

    def test_target_mismatch_fails_before_builder_dispatch(self, tmp_path: Path):
        from mobius.integrations.gguf._mmproj import build_vlm_from_gguf

        text_path = tmp_path / "muse-glimmer.gguf"
        mmproj_path = tmp_path / "mmproj.gguf"
        identity_repo = "https://huggingface.co/example/shared-vlm"
        _write_minimal_gguf(
            text_path,
            "muse-glimmer",
            identity_name="Shared-VLM",
            identity_repo=identity_repo,
        )
        _write_clip_mmproj_gguf(
            mmproj_path,
            identity_name="Shared-VLM",
            identity_repo=identity_repo,
        )

        with (
            mock.patch(
                "mobius.integrations.gguf._mmproj.build_gemma4_vlm_from_gguf"
            ) as builder,
            pytest.raises(ValueError, match=r"targets.*gemma4"),
        ):
            build_vlm_from_gguf(text_path, mmproj_path)
        builder.assert_not_called()

    def test_scale_tensor_cannot_be_silently_dropped(self, tmp_path: Path):
        from mobius.integrations.gguf._mmproj import build_gemma4_vlm_from_gguf

        text_path = tmp_path / "gemma4.gguf"
        mmproj_path = tmp_path / "mmproj.gguf"
        _write_minimal_gguf(text_path, "gemma4")
        _write_clip_mmproj_gguf(mmproj_path, extra_tensor="v.blk.0.attn_q.input_scale")

        with (
            mock.patch("mobius._builder.build_from_module") as build_graph,
            pytest.raises(ValueError, match=r"input_scale.*never dropped"),
        ):
            build_gemma4_vlm_from_gguf(text_path, mmproj_path, image_token_id=0)
        build_graph.assert_not_called()

    @pytest.mark.parametrize(
        "extra_tensor",
        ["unexpected.weight", "vision.blk.0.attn_q.weight"],
    )
    def test_complete_inventory_rejects_unknown_tensor_names(
        self, tmp_path: Path, extra_tensor: str
    ):
        from mobius.integrations.gguf._mmproj import _preflight_mmproj_pair
        from mobius.integrations.gguf._mmproj_registry import MMProjModality

        text_path = tmp_path / "gemma4.gguf"
        mmproj_path = tmp_path / "mmproj.gguf"
        _write_minimal_gguf(text_path, "gemma4")
        _write_clip_mmproj_gguf(mmproj_path, with_audio=False, extra_tensor=extra_tensor)
        text, mmproj = self._load_pair(text_path, mmproj_path)

        with pytest.raises(ValueError, match=r"outside the pinned.*closure"):
            _preflight_mmproj_pair(text, mmproj, modalities=(MMProjModality.VISION,))

    def test_valid_deferred_audio_companion_inventory_is_explicitly_allowed(
        self, tmp_path: Path
    ):
        from mobius.integrations.gguf._mmproj import _preflight_mmproj_pair
        from mobius.integrations.gguf._mmproj_registry import MMProjModality

        text_path = tmp_path / "gemma4.gguf"
        mmproj_path = tmp_path / "mmproj.gguf"
        _write_minimal_gguf(text_path, "gemma4")
        _write_clip_mmproj_gguf(mmproj_path, with_audio=True)
        text, mmproj = self._load_pair(text_path, mmproj_path)

        resolved = _preflight_mmproj_pair(text, mmproj, modalities=(MMProjModality.VISION,))
        assert resolved[MMProjModality.VISION].projector_type == "gemma4v"

    def test_companion_tensor_wrong_rank_is_rejected(self, tmp_path: Path):
        from mobius.integrations.gguf._mmproj import _preflight_mmproj_pair
        from mobius.integrations.gguf._mmproj_registry import MMProjModality

        text_path = tmp_path / "gemma4.gguf"
        mmproj_path = tmp_path / "mmproj.gguf"
        _write_minimal_gguf(text_path, "gemma4")
        _write_clip_mmproj_gguf(mmproj_path, with_audio=True)
        text, mmproj = self._load_pair(text_path, mmproj_path)
        original = mmproj.get_tensor_shape

        with (
            mock.patch.object(
                mmproj,
                "get_tensor_shape",
                side_effect=lambda name: (
                    (1, 3, 3) if name == "a.conv1d.0.weight" else original(name)
                ),
            ),
            pytest.raises(ValueError, match=r"a\.conv1d\.0\.weight must have shape"),
        ):
            _preflight_mmproj_pair(text, mmproj, modalities=(MMProjModality.VISION,))

    @pytest.mark.parametrize("missing_side", ["text", "mmproj"])
    def test_missing_identity_on_either_side_fails_closed(
        self, tmp_path: Path, missing_side: str
    ):
        from mobius.integrations.gguf._mmproj import _preflight_mmproj_pair
        from mobius.integrations.gguf._mmproj_registry import MMProjModality

        text_path = tmp_path / "gemma4.gguf"
        mmproj_path = tmp_path / "mmproj.gguf"
        _write_minimal_gguf(text_path, "gemma4")
        _write_clip_mmproj_gguf(mmproj_path, with_audio=False)
        text, mmproj = self._load_pair(text_path, mmproj_path)
        (text if missing_side == "text" else mmproj).metadata.pop("general.name")

        with pytest.raises(ValueError, match=r"both files must declare.*general\.name"):
            _preflight_mmproj_pair(text, mmproj, modalities=(MMProjModality.VISION,))

    @pytest.mark.parametrize(
        ("key", "mismatched"),
        [
            ("general.name", "Other-Gemma"),
            (
                "general.base_model.0.repo_url",
                "https://huggingface.co/example/other-gemma",
            ),
        ],
    )
    def test_identity_binding_mismatch_is_rejected(
        self, tmp_path: Path, key: str, mismatched: str
    ):
        from mobius.integrations.gguf._mmproj import _preflight_mmproj_pair
        from mobius.integrations.gguf._mmproj_registry import MMProjModality

        text_path = tmp_path / "gemma4.gguf"
        mmproj_path = tmp_path / "mmproj.gguf"
        _write_minimal_gguf(text_path, "gemma4")
        _write_clip_mmproj_gguf(mmproj_path, with_audio=False)
        text, mmproj = self._load_pair(text_path, mmproj_path)
        mmproj.metadata[key] = mismatched

        with pytest.raises(ValueError, match=key):
            _preflight_mmproj_pair(text, mmproj, modalities=(MMProjModality.VISION,))

    def test_matching_identity_survives_file_relocation(self, tmp_path: Path):
        from mobius.integrations.gguf._mmproj import _preflight_mmproj_pair
        from mobius.integrations.gguf._mmproj_registry import MMProjModality

        text_path = tmp_path / "relocated" / "text" / "renamed-model.gguf"
        mmproj_path = tmp_path / "another-root" / "renamed-sidecar.gguf"
        text_path.parent.mkdir(parents=True)
        mmproj_path.parent.mkdir(parents=True)
        _write_minimal_gguf(text_path, "gemma4")
        _write_clip_mmproj_gguf(mmproj_path, with_audio=False)
        text, mmproj = self._load_pair(text_path, mmproj_path)

        resolved = _preflight_mmproj_pair(text, mmproj, modalities=(MMProjModality.VISION,))
        assert resolved[MMProjModality.VISION].projector_type == "gemma4v"

    @pytest.mark.parametrize(
        ("text_name", "sidecar_name"),
        [
            ("Model-A", "ModelA"),
            ("org/model", "orgmodel"),
            ("model_a", "model-a"),
        ],
    )
    def test_identity_normalization_preserves_meaningful_separators(
        self, tmp_path: Path, text_name: str, sidecar_name: str
    ):
        from mobius.integrations.gguf._mmproj import _preflight_mmproj_pair
        from mobius.integrations.gguf._mmproj_registry import MMProjModality

        text_path = tmp_path / "gemma4.gguf"
        mmproj_path = tmp_path / "mmproj.gguf"
        _write_minimal_gguf(text_path, "gemma4")
        _write_clip_mmproj_gguf(mmproj_path, with_audio=False)
        text, mmproj = self._load_pair(text_path, mmproj_path)
        text.metadata["general.name"] = text_name
        mmproj.metadata["general.name"] = sidecar_name

        with pytest.raises(ValueError, match=r"general\.name"):
            _preflight_mmproj_pair(text, mmproj, modalities=(MMProjModality.VISION,))

    @pytest.mark.parametrize("empty_value", ["", " \t "])
    def test_empty_identity_values_are_rejected(self, tmp_path: Path, empty_value: str):
        from mobius.integrations.gguf._mmproj import _preflight_mmproj_pair
        from mobius.integrations.gguf._mmproj_registry import MMProjModality

        text_path = tmp_path / "gemma4.gguf"
        mmproj_path = tmp_path / "mmproj.gguf"
        _write_minimal_gguf(text_path, "gemma4")
        _write_clip_mmproj_gguf(mmproj_path, with_audio=False)
        text, mmproj = self._load_pair(text_path, mmproj_path)
        text.metadata["general.name"] = empty_value
        mmproj.metadata["general.name"] = empty_value

        with pytest.raises(ValueError, match=r"non-empty string"):
            _preflight_mmproj_pair(text, mmproj, modalities=(MMProjModality.VISION,))

    def test_one_sided_present_empty_optional_binding_is_not_treated_absent(
        self, tmp_path: Path
    ):
        from mobius.integrations.gguf._mmproj import _preflight_mmproj_pair
        from mobius.integrations.gguf._mmproj_registry import MMProjModality

        text_path = tmp_path / "gemma4.gguf"
        mmproj_path = tmp_path / "mmproj.gguf"
        _write_minimal_gguf(text_path, "gemma4")
        _write_clip_mmproj_gguf(mmproj_path, with_audio=False)
        text, mmproj = self._load_pair(text_path, mmproj_path)
        text.metadata["general.base_model.0.repo_url"] = ""
        mmproj.metadata.pop("general.base_model.0.repo_url")

        with pytest.raises(ValueError, match=r"present on only one file"):
            _preflight_mmproj_pair(text, mmproj, modalities=(MMProjModality.VISION,))

    def test_repository_canonicalization_only_normalizes_url_syntax(self, tmp_path: Path):
        from mobius.integrations.gguf._mmproj import _preflight_mmproj_pair
        from mobius.integrations.gguf._mmproj_registry import MMProjModality

        text_path = tmp_path / "gemma4.gguf"
        mmproj_path = tmp_path / "mmproj.gguf"
        _write_minimal_gguf(text_path, "gemma4")
        _write_clip_mmproj_gguf(mmproj_path, with_audio=False)
        text, mmproj = self._load_pair(text_path, mmproj_path)
        text.metadata["general.base_model.0.repo_url"] = (
            "HTTPS://HUGGINGFACE.CO/Google/Gemma-4-E2B-It.git/"
        )

        resolved = _preflight_mmproj_pair(text, mmproj, modalities=(MMProjModality.VISION,))
        assert resolved[MMProjModality.VISION].projector_type == "gemma4v"

    @pytest.mark.parametrize(
        "patch_shape",
        [
            (_VISION_HIDDEN, 1, 3, _PATCH_SIZE, _PATCH_SIZE),
            (_VISION_HIDDEN, 1, _PATCH_SIZE, _PATCH_SIZE),
            (_VISION_HIDDEN, 3, _PATCH_SIZE, _PATCH_SIZE + 1),
        ],
    )
    def test_gemma4_patch_shape_rejects_before_graph_build(
        self, tmp_path: Path, patch_shape: tuple[int, ...]
    ):
        from mobius.integrations.gguf._mmproj import build_gemma4_vlm_from_gguf

        text_path = tmp_path / "gemma4.gguf"
        mmproj_path = tmp_path / "mmproj.gguf"
        _write_minimal_gguf(text_path, "gemma4")
        _write_clip_mmproj_gguf(
            mmproj_path,
            with_audio=False,
            patch_weight_shape=patch_shape,
        )

        with (
            mock.patch("mobius._builder.build_from_module") as build_graph,
            pytest.raises(ValueError, match=r"graph-compatible shape"),
        ):
            build_gemma4_vlm_from_gguf(text_path, mmproj_path, image_token_id=0)
        build_graph.assert_not_called()

    @pytest.mark.parametrize("num_mel_bins", [None, 0])
    def test_audio_num_mel_bins_rejects_before_graph_build(
        self, tmp_path: Path, num_mel_bins: int | None
    ):
        from mobius.integrations.gguf._mmproj import build_gemma4_vlm_from_gguf

        text_path = tmp_path / "gemma4.gguf"
        mmproj_path = tmp_path / "mmproj.gguf"
        _write_minimal_gguf(text_path, "gemma4")
        _write_clip_mmproj_gguf(
            mmproj_path,
            with_audio=True,
            audio_num_mel_bins=num_mel_bins,
        )

        with (
            mock.patch("mobius._builder.build_from_module") as build_graph,
            pytest.raises(ValueError, match=r"clip\.audio\.num_mel_bins"),
        ):
            build_gemma4_vlm_from_gguf(text_path, mmproj_path, image_token_id=0)
        build_graph.assert_not_called()

    def test_audio_num_mel_bins_wrong_type_is_rejected(self, tmp_path: Path):
        from mobius.integrations.gguf._mmproj import _preflight_mmproj_pair
        from mobius.integrations.gguf._mmproj_registry import MMProjModality

        text_path = tmp_path / "gemma4.gguf"
        mmproj_path = tmp_path / "mmproj.gguf"
        _write_minimal_gguf(text_path, "gemma4")
        _write_clip_mmproj_gguf(mmproj_path, with_audio=True)
        text, mmproj = self._load_pair(text_path, mmproj_path)
        mmproj.metadata["clip.audio.num_mel_bins"] = "8"

        with pytest.raises(ValueError, match=r"positive integer"):
            _preflight_mmproj_pair(text, mmproj, modalities=(MMProjModality.VISION,))

    def test_quantized_projector_weight_is_rejected_by_role(self, tmp_path: Path):
        from types import SimpleNamespace

        from mobius.integrations.gguf._mmproj import (
            _preflight_mmproj_pair,
        )
        from mobius.integrations.gguf._mmproj_registry import MMProjModality
        from mobius.integrations.gguf._reader import GGUFModel

        text_path = tmp_path / "gemma4.gguf"
        mmproj_path = tmp_path / "mmproj.gguf"
        _write_minimal_gguf(text_path, "gemma4")
        _write_clip_mmproj_gguf(mmproj_path)
        text = GGUFModel(str(text_path))
        mmproj = GGUFModel(str(mmproj_path))
        original = mmproj.get_tensor_type

        def tensor_type(name: str):
            if name == "mm.input_projection.weight":
                return SimpleNamespace(name="Q8_0")
            return original(name)

        with (
            mock.patch.object(mmproj, "get_tensor_type", side_effect=tensor_type),
            pytest.raises(NotImplementedError, match="packed Q8_0"),
        ):
            _preflight_mmproj_pair(text, mmproj, modalities=(MMProjModality.VISION,))


class TestReadVisionConfig:
    def test_extracts_expected_fields(self, clip_mmproj_gguf: Path):
        from mobius.integrations.gguf._mmproj import read_mmproj_vision_config
        from mobius.integrations.gguf._reader import GGUFModel

        config = read_mmproj_vision_config(GGUFModel(str(clip_mmproj_gguf)))

        assert config is not None
        assert config.hidden_size == _VISION_HIDDEN
        assert config.intermediate_size == _VISION_FFN
        assert config.num_hidden_layers == _VISION_LAYERS
        assert config.num_attention_heads == _VISION_HEADS
        assert config.image_size == _IMAGE_SIZE
        assert config.patch_size == _PATCH_SIZE
        assert config.position_embedding_size == _POS_EMB_SIZE
        assert config.use_clipped_linears is True
        assert config.pooling_kernel_size == 3
        assert config.norm_eps == pytest.approx(1e-6)

    def test_returns_none_without_vision_encoder(self, tmp_path: Path):
        from mobius.integrations.gguf._mmproj import read_mmproj_vision_config
        from mobius.integrations.gguf._reader import GGUFModel

        path = tmp_path / "audio_only.gguf"
        _write_clip_mmproj_gguf(path, with_audio=True)
        model = GGUFModel(str(path))
        model.metadata["clip.has_vision_encoder"] = False
        assert read_mmproj_vision_config(model) is None


class TestReadAudioConfig:
    def test_extracts_expected_fields(self, clip_mmproj_gguf: Path):
        from mobius.integrations.gguf._mmproj import read_mmproj_audio_config
        from mobius.integrations.gguf._reader import GGUFModel

        config = read_mmproj_audio_config(GGUFModel(str(clip_mmproj_gguf)))

        assert config is not None
        assert config.hidden_size == _AUDIO_HIDDEN
        assert config.num_layers == _AUDIO_LAYERS
        assert config.attention_heads == _AUDIO_HEADS
        assert config.input_size == _NUM_MEL_BINS
        assert config.subsampling_conv_channels == [_AUDIO_CONV0, _AUDIO_CONV1]
        assert config.output_proj_dims == _AUDIO_PROJ_OUT

    def test_returns_none_without_audio_encoder(self, tmp_path: Path):
        from mobius.integrations.gguf._mmproj import read_mmproj_audio_config
        from mobius.integrations.gguf._reader import GGUFModel

        path = tmp_path / "vision_only.gguf"
        _write_clip_mmproj_gguf(path, with_audio=False)
        assert read_mmproj_audio_config(GGUFModel(str(path))) is None


class TestVisionEncoderBuildAndRun:
    """Build the Gemma4 vision encoder from the synthetic mmproj and run it."""

    def test_declares_single_image_batch_contract(self, clip_mmproj_gguf: Path):
        from mobius._configs import Gemma4Config
        from mobius.integrations.gguf._mmproj import read_mmproj_vision_config
        from mobius.integrations.gguf._reader import GGUFModel
        from mobius.models.gemma4 import _Gemma4VisionEncoderModel
        from mobius.tasks._gemma4 import Gemma4Task

        vision_config = read_mmproj_vision_config(GGUFModel(str(clip_mmproj_gguf)))
        config = Gemma4Config(
            hidden_size=_TEXT_HIDDEN,
            num_hidden_layers=1,
            num_attention_heads=2,
            vocab_size=64,
            vision=vision_config,
        )
        model = Gemma4Task()._build_vision(_Gemma4VisionEncoderModel(config), config)

        assert next(iter(model.graph.inputs[0].shape)) == 1
        assert next(iter(model.graph.inputs[1].shape)) == 1

    def test_matches_independent_numpy_reference(self, tmp_path: Path):
        """Check patch, position, pooling, norm, and projection semantics."""
        import onnx_ir as ir
        import onnxruntime as ort

        from mobius._configs import Gemma4Config, VisionConfig
        from mobius.models.gemma4 import _Gemma4VisionEncoderModel
        from mobius.tasks._gemma4 import Gemma4Task

        rng = np.random.default_rng(41)
        vision = VisionConfig(
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=0,
            num_attention_heads=2,
            image_size=2,
            patch_size=1,
            position_embedding_size=2,
            position_embedding_height=2,
            position_embedding_width=2,
            pooling_kernel_size=1,
            use_clipped_linears=True,
        )
        config = Gemma4Config(
            hidden_size=6,
            num_hidden_layers=1,
            num_attention_heads=2,
            vocab_size=8,
            vision=vision,
        )
        model = Gemma4Task()._build_vision(_Gemma4VisionEncoderModel(config), config)
        weights = {
            "encoder.patch_embedder.position_embedding_table": rng.normal(
                size=(2, 2, 4)
            ).astype(np.float32),
            "encoder.patch_embedder.input_proj.weight": rng.normal(size=(4, 3)).astype(
                np.float32
            ),
            "projector_norm.weight": np.ones(4, dtype=np.float32),
            "projector.weight": rng.normal(size=(6, 4)).astype(np.float32),
        }
        for name, value in weights.items():
            model.graph.initializers[name].const_value = ir.tensor(value)

        model_path = tmp_path / "gemma4-parity.onnx"
        ir.save(model, str(model_path))
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        pixel_values = rng.uniform(size=(1, 4, 3)).astype(np.float32)
        positions = np.array([[[0, 0], [1, 0], [0, 1], [1, 1]]], dtype=np.int64)
        actual = session.run(
            None,
            {"pixel_values": pixel_values, "pixel_position_ids": positions},
        )[0]

        projected = (2.0 * pixel_values - 1.0) @ weights[
            "encoder.patch_embedder.input_proj.weight"
        ].T
        projected += (
            weights["encoder.patch_embedder.position_embedding_table"][0, positions[..., 0]]
            + weights["encoder.patch_embedder.position_embedding_table"][1, positions[..., 1]]
        )
        pooled = projected * np.sqrt(4.0)
        normalized = pooled / np.sqrt(
            np.mean(np.square(pooled), axis=-1, keepdims=True) + vision.norm_eps
        )
        expected = normalized @ weights["projector.weight"].T
        np.testing.assert_allclose(actual, expected.reshape(4, 6), rtol=1e-5, atol=1e-5)

    def test_builds_applies_and_runs(self, clip_mmproj_gguf: Path, tmp_path: Path):
        import onnxruntime as ort

        from mobius._configs import Gemma4Config
        from mobius._model_package import ModelPackage
        from mobius.integrations.gguf._mmproj import (
            _mmproj_vision_to_hf,
            read_mmproj_vision_config,
        )
        from mobius.integrations.gguf._reader import GGUFModel
        from mobius.models.gemma4 import _Gemma4VisionEncoderModel
        from mobius.tasks._gemma4 import Gemma4Task

        gguf_model = GGUFModel(str(clip_mmproj_gguf))
        vision_config = read_mmproj_vision_config(gguf_model)
        config = Gemma4Config(
            hidden_size=_TEXT_HIDDEN,
            num_hidden_layers=1,
            num_attention_heads=2,
            vocab_size=64,
            vision=vision_config,
        )

        module = _Gemma4VisionEncoderModel(config)
        model = Gemma4Task()._build_vision(module, config)
        package = ModelPackage({"vision_encoder": model}, config=config)

        # Map mmproj vision tensors → HF names, then through the module's own
        # preprocessing (vision_tower.* / embed_vision.* → module params).
        state_dict = _mmproj_vision_to_hf(gguf_model)
        state_dict = module.preprocess_weights(state_dict)
        package.apply_weights(state_dict)

        out_dir = tmp_path / "vision_out"
        package.save(str(out_dir), progress_bar=False)

        session = ort.InferenceSession(
            str(out_dir / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        grid = _IMAGE_SIZE // _PATCH_SIZE
        num_patches = grid * grid
        pixel_values = np.random.rand(1, num_patches, 3 * _PATCH_SIZE * _PATCH_SIZE).astype(
            np.float32
        )
        xs, ys = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
        pixel_position_ids = np.stack([xs.ravel(), ys.ravel()], axis=-1)[None].astype(np.int64)

        outputs = session.run(
            None,
            {"pixel_values": pixel_values, "pixel_position_ids": pixel_position_ids},
        )
        image_features = outputs[0]
        assert image_features.ndim == 2
        assert image_features.shape[1] == _TEXT_HIDDEN
        assert np.isfinite(image_features).all()


def _component_op_types(model) -> set[str]:
    """Collect every ONNX op type used across a component's graph."""
    return {node.op_type for node in model.graph}


class TestKeepQuantizedMixedPrecision:
    """A ``keep_quantized`` multimodal build must be mixed precision.

    The Gemma4 *text* decoder + token embedding honour ``config.quantization``
    (MatMulNBits / GatherBlockQuantized), while the mmproj-sourced vision
    encoder stays float — see ``build_gemma4_vlm_from_gguf``'s "Mixed
    precision" note. This asserts that property directly on the built graphs,
    using the same ``Gemma4Model`` + ``Gemma4Task`` build path the multimodal
    builder uses, without needing a full quantized text GGUF.
    """

    def test_multimodal_builder_preserves_quantization_by_default(self):
        import inspect

        from mobius.integrations.gguf import build_gemma4_vlm_from_gguf

        parameter = inspect.signature(build_gemma4_vlm_from_gguf).parameters["keep_quantized"]
        assert parameter.default is True

    def test_decoder_and_embedding_quantized_vision_stays_float(self, clip_mmproj_gguf: Path):
        import dataclasses

        from mobius._configs import Gemma4Config, QuantizationConfig
        from mobius.integrations.gguf._mmproj import read_mmproj_vision_config
        from mobius.integrations.gguf._reader import GGUFModel
        from mobius.models.gemma4 import Gemma4Model
        from mobius.tasks._gemma4 import Gemma4Task

        vision_config = read_mmproj_vision_config(GGUFModel(str(clip_mmproj_gguf)))
        assert vision_config is not None

        config = Gemma4Config(
            hidden_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=8,
            global_head_dim=16,
            num_global_key_value_heads=1,
            vocab_size=64,
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
            ],
            num_kv_shared_layers=1,
            hidden_size_per_layer_input=8,
            vocab_size_per_layer_input=64,
            intermediate_size=64,
            hidden_act="gelu_pytorch_tanh",
            vision=vision_config,
        )
        # A single module-global quantization config: only the Gemma4 text
        # components read it; the vision encoder ignores it and stays float.
        config = dataclasses.replace(
            config,
            quantization=QuantizationConfig(
                bits=4,
                group_size=32,
                quant_method="gguf",
                sym=False,
                quantize_embeddings=True,
                quantize_lm_head=True,
            ),
        )

        package = Gemma4Task().build(Gemma4Model(config), config)
        assert set(package) == {"decoder", "vision_encoder", "embedding"}

        decoder_ops = _component_op_types(package["decoder"])
        embedding_ops = _component_op_types(package["embedding"])
        vision_ops = _component_op_types(package["vision_encoder"])

        # Text decoder projections are packed 4-bit matmuls.
        assert "MatMulNBits" in decoder_ops
        # Token-embedding table stays packed (int4 gather).
        assert "GatherBlockQuantized" in embedding_ops
        # The mmproj-sourced vision encoder is float — no quantized ops.
        assert "MatMulNBits" not in vision_ops
        assert "GatherBlockQuantized" not in vision_ops

    def test_incompatible_embedding_stays_float_and_package_round_trips(
        self,
        clip_mmproj_gguf: Path,
        tmp_path: Path,
    ):
        from mobius._model_package import ModelPackage
        from mobius.integrations.gguf import build_gemma4_vlm_from_gguf

        text_gguf = tmp_path / "gemma4-q4-f32-embedding.gguf"
        _write_quantized_gemma4_text_gguf(text_gguf)

        package = build_gemma4_vlm_from_gguf(text_gguf, clip_mmproj_gguf, image_token_id=63)
        assert "MatMulNBits" in _component_op_types(package["decoder"])
        assert "GatherBlockQuantized" not in _component_op_types(package["embedding"])
        float_embedding = package["embedding"].graph.initializers[
            "embedding.embed_tokens.weight"
        ]
        assert float_embedding.const_value is not None
        assert list(float_embedding.shape) == [64, 32]

        missing = [
            f"{component}:{name}"
            for component, model in package.items()
            for name, initializer in model.graph.initializers.items()
            if initializer.const_value is None
        ]
        assert missing == []

        output_dir = tmp_path / "saved"
        package.save(str(output_dir), progress_bar=False)
        reloaded = ModelPackage.load(str(output_dir))
        assert set(reloaded) == {"decoder", "vision_encoder", "embedding"}
        assert all(
            initializer.const_value is not None
            for model in reloaded.values()
            for initializer in model.graph.initializers.values()
        )

    def test_quantized_decoder_loads_in_onnxruntime(
        self, clip_mmproj_gguf: Path, tmp_path: Path
    ):
        """The quantized decoder (incl. KV-shared layers) loads in ORT.

        Session creation runs onnxruntime's shape inference, which is where the
        pre-fix graph failed for KV-shared layers with a MatMul "Incompatible
        dimensions" error. This guards the fix that gives KV-shared layers'
        shared K/V a static hidden size so onnxruntime can shape-infer the
        attention output width.
        """
        import dataclasses

        import onnx_ir as ir
        import onnxruntime as ort

        from mobius._configs import Gemma4Config, QuantizationConfig
        from mobius.integrations.gguf._mmproj import read_mmproj_vision_config
        from mobius.integrations.gguf._reader import GGUFModel
        from mobius.models.gemma4 import _Gemma4DecoderModel
        from mobius.tasks._gemma4 import Gemma4Task

        vision_config = read_mmproj_vision_config(GGUFModel(str(clip_mmproj_gguf)))
        config = Gemma4Config(
            hidden_size=32,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=8,
            global_head_dim=16,
            num_global_key_value_heads=1,
            vocab_size=64,
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
            num_kv_shared_layers=2,
            hidden_size_per_layer_input=8,
            vocab_size_per_layer_input=64,
            intermediate_size=64,
            hidden_act="gelu_pytorch_tanh",
            vision=vision_config,
        )
        config = dataclasses.replace(
            config,
            quantization=QuantizationConfig(
                bits=4,
                group_size=32,
                quant_method="gguf",
                sym=False,
                quantize_embeddings=True,
                quantize_lm_head=True,
            ),
        )

        model = Gemma4Task()._build_decoder(_Gemma4DecoderModel(config), config)
        # Fill random data so the graph can be serialized and loaded by ORT.
        for init in model.graph.initializers.values():
            if init.const_value is not None:
                continue
            shape = tuple(d if isinstance(d, int) else 1 for d in init.shape)
            np_dtype = init.dtype.numpy() if init.dtype is not None else np.float32
            if np.issubdtype(np_dtype, np.integer):
                values = np.random.randint(0, 7, size=shape).astype(np_dtype)
            else:
                values = (np.random.rand(*shape).astype(np.float32) * 0.1).astype(np_dtype)
            init.const_value = ir.tensor(values)

        decoder_path = tmp_path / "decoder.onnx"
        ir.save(model, str(decoder_path))

        # Session creation runs shape inference — the KV-shared fix is what keeps
        # this from failing with a MatMul "Incompatible dimensions" error.
        session = ort.InferenceSession(str(decoder_path), providers=["CPUExecutionProvider"])
        input_names = {graph_input.name for graph_input in session.get_inputs()}
        assert "inputs_embeds" in input_names


# Small synthetic Muse Glimmer vision tower dimensions.
_MG_HIDDEN = 16
_MG_FFN = 32
_MG_LAYERS = 6
_MG_HEADS = 2
_MG_PATCH = 4
_MG_GRID = 4
_MG_MERGE = 2
_MG_PROJECTOR = 24
_MG_TEXT_HIDDEN = 32


def _write_muse_glimmer_mmproj_gguf(path: Path) -> None:
    """Write a small synthetic Muse Glimmer ``clip`` mmproj GGUF.

    Mirrors the published ``mmproj-Muse-Glimmer-30B-*.gguf`` layout: biased
    projections, a plain two-LayerNorm block, no SwiGLU gate, no QK norms, and
    three positional ``mm.N`` projector matrices.
    """
    from gguf import GGUFWriter

    writer = GGUFWriter(str(path), "clip")
    writer.add_string("general.name", "Muse-Glimmer-30B")
    writer.add_string("general.type", "mmproj")
    writer.add_bool("clip.has_vision_encoder", True)
    writer.add_string("clip.projector_type", "muse-glimmer")
    writer.add_uint32("clip.vision.embedding_length", _MG_HIDDEN)
    writer.add_uint32("clip.vision.feed_forward_length", _MG_FFN)
    writer.add_uint32("clip.vision.block_count", _MG_LAYERS)
    writer.add_uint32("clip.vision.attention.head_count", _MG_HEADS)
    writer.add_uint32("clip.vision.image_size", 64)
    writer.add_uint32("clip.vision.patch_size", _MG_PATCH)
    writer.add_uint32("clip.vision.projection_dim", _MG_TEXT_HIDDEN)
    writer.add_uint32("clip.vision.spatial_merge_size", _MG_MERGE)
    writer.add_array("clip.vision.image_mean", [0.5, 0.5, 0.5])
    writer.add_array("clip.vision.image_std", [0.5, 0.5, 0.5])
    writer.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-5)

    def _f32(name: str, shape: tuple[int, ...]) -> None:
        writer.add_tensor(name, np.random.randn(*shape).astype(np.float32))

    _f32("v.patch_embd.weight", (_MG_HIDDEN, 3, _MG_PATCH, _MG_PATCH))
    _f32("v.position_embd.weight", (_MG_GRID * _MG_GRID, _MG_HIDDEN))
    for stem in ("v.pre_ln", "v.post_ln"):
        _f32(f"{stem}.weight", (_MG_HIDDEN,))
        _f32(f"{stem}.bias", (_MG_HIDDEN,))
    shuffled = _MG_HIDDEN * _MG_MERGE * _MG_MERGE
    _f32("mm.0.weight", (_MG_PROJECTOR, shuffled))
    _f32("mm.1.weight", (_MG_PROJECTOR, _MG_PROJECTOR))
    _f32("mm.2.weight", (_MG_TEXT_HIDDEN, _MG_PROJECTOR))

    for layer in range(_MG_LAYERS):
        prefix = f"v.blk.{layer}."
        for norm in ("ln1", "ln2"):
            _f32(prefix + norm + ".weight", (_MG_HIDDEN,))
            _f32(prefix + norm + ".bias", (_MG_HIDDEN,))
        for proj in ("attn_q", "attn_k", "attn_v", "attn_out"):
            _f32(prefix + proj + ".weight", (_MG_HIDDEN, _MG_HIDDEN))
            _f32(prefix + proj + ".bias", (_MG_HIDDEN,))
        _f32(prefix + "ffn_up.weight", (_MG_FFN, _MG_HIDDEN))
        _f32(prefix + "ffn_up.bias", (_MG_FFN,))
        _f32(prefix + "ffn_down.weight", (_MG_HIDDEN, _MG_FFN))
        _f32(prefix + "ffn_down.bias", (_MG_HIDDEN,))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class TestReadMuseGlimmerVisionConfig:
    """Muse Glimmer vision config extraction from a ``clip`` mmproj."""

    @pytest.fixture
    def mmproj(self, tmp_path: Path) -> Path:
        path = tmp_path / "mmproj-muse.gguf"
        _write_muse_glimmer_mmproj_gguf(path)
        return path

    def test_reads_metadata_fields(self, mmproj: Path):
        from mobius.integrations.gguf._mmproj import (
            read_mmproj_muse_glimmer_vision_config,
        )
        from mobius.integrations.gguf._reader import GGUFModel

        config = read_mmproj_muse_glimmer_vision_config(GGUFModel(str(mmproj)))

        assert config is not None
        assert config.hidden_size == _MG_HIDDEN
        assert config.intermediate_size == _MG_FFN
        assert config.num_hidden_layers == _MG_LAYERS
        assert config.num_attention_heads == _MG_HEADS
        assert config.patch_size == _MG_PATCH
        assert config.spatial_merge_size == _MG_MERGE
        assert config.in_channels == 3
        assert config.norm_eps == pytest.approx(1e-5)
        assert config.hidden_act == "gelu"

    def test_recovers_the_fields_gguf_does_not_store(self, mmproj: Path):
        from mobius.integrations.gguf._mmproj import (
            read_mmproj_muse_glimmer_vision_config,
        )
        from mobius.integrations.gguf._reader import GGUFModel

        config = read_mmproj_muse_glimmer_vision_config(GGUFModel(str(mmproj)))

        assert config is not None
        # Square position grid, read back from the table's row count.
        assert config.position_embedding_size == _MG_GRID * _MG_GRID
        assert config.position_embedding_height == _MG_GRID
        assert config.position_embedding_width == _MG_GRID
        # Adapter width comes from mm.0.
        assert config.projector_intermediate_size == _MG_PROJECTOR
        # HF's out_hidden_size is the pixel-shuffled width, not the text width.
        assert config.out_hidden_size == _MG_HIDDEN * _MG_MERGE * _MG_MERGE
        # Every 4th block is global attention, and so is the last one.
        assert config.fullatt_block_indexes == [3, 5]
        # Derived from the patch-embedding weight: llama.cpp stores a plain
        # Conv2d kernel, i.e. a single temporal frame.
        assert config.temporal_patch_size == 1
        assert config.rope_theta == pytest.approx(10_000.0)

    def test_returns_none_without_vision_encoder(self, mmproj: Path):
        from mobius.integrations.gguf._mmproj import (
            read_mmproj_muse_glimmer_vision_config,
        )
        from mobius.integrations.gguf._reader import GGUFModel

        model = GGUFModel(str(mmproj))
        model.metadata["clip.has_vision_encoder"] = False
        assert read_mmproj_muse_glimmer_vision_config(model) is None

    def test_non_square_position_table_is_rejected(self, mmproj: Path):
        from mobius.integrations.gguf._mmproj import (
            read_mmproj_muse_glimmer_vision_config,
        )
        from mobius.integrations.gguf._reader import GGUFModel

        model = GGUFModel(str(mmproj))
        with (
            mock.patch.object(
                model,
                "get_tensor",
                side_effect=lambda name: np.zeros((15, _MG_HIDDEN), dtype=np.float32),
            ),
            pytest.raises(ValueError, match="not a square grid"),
        ):
            read_mmproj_muse_glimmer_vision_config(model)

    def test_vision_tensors_load_under_hf_names(self, mmproj: Path):
        from mobius.integrations.gguf._mmproj import _mmproj_muse_glimmer_vision_to_hf
        from mobius.integrations.gguf._reader import GGUFModel

        state = _mmproj_muse_glimmer_vision_to_hf(GGUFModel(str(mmproj)))

        # The conv patch embedding is flattened into the encoder's Linear.
        patch = state["model.vision_tower.patch_embedder.patch_embedding.weight"]
        assert tuple(patch.shape) == (_MG_HIDDEN, 3 * _MG_PATCH * _MG_PATCH)
        assert tuple(state["model.vision_projection.weight"].shape) == (
            _MG_TEXT_HIDDEN,
            _MG_PROJECTOR,
        )
        assert "model.vision_tower.layers.5.attn.proj.bias" in state
        # 6 blocks x 16 tensors + 6 stem tensors + 3 projector matrices.
        assert len(state) == _MG_LAYERS * 16 + 6 + 3


class TestMuseGlimmerVisionEncoder:
    """Numerical checks for the supported Muse Glimmer encoder/projector path."""

    def test_matches_independent_numpy_reference(self, tmp_path: Path):
        import math

        import onnx_ir as ir
        import onnxruntime as ort

        from mobius._configs import MuseGlimmerConfig, VisionConfig
        from mobius.models.muse_glimmer import MuseGlimmerVisionEncoderModel
        from mobius.tasks import MuseGlimmerVLTask

        rng = np.random.default_rng(42)
        vision = VisionConfig(
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=0,
            num_attention_heads=2,
            image_size=2,
            patch_size=1,
            position_embedding_height=2,
            position_embedding_width=2,
            spatial_merge_size=1,
            temporal_patch_size=1,
            in_channels=3,
            projector_intermediate_size=4,
            fullatt_block_indexes=[],
        )
        config = MuseGlimmerConfig(
            hidden_size=6,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=3,
            vocab_size=8,
            intermediate_size=8,
            hidden_act="gelu",
            temporal_patch_size=1,
            vision=vision,
        )
        model = MuseGlimmerVLTask()._build_vision(
            MuseGlimmerVisionEncoderModel(config), config
        )
        weights = {
            "vision_tower.patch_embedder.position_embedding_table.weight": rng.normal(
                size=(4, 4)
            ).astype(np.float32),
            "vision_tower.patch_embedder.patch_embedding.weight": rng.normal(
                size=(4, 3)
            ).astype(np.float32),
            "vision_tower.ln_pre.weight": rng.normal(size=4).astype(np.float32),
            "vision_tower.ln_pre.bias": rng.normal(size=4).astype(np.float32),
            "vision_tower.ln_post.weight": rng.normal(size=4).astype(np.float32),
            "vision_tower.ln_post.bias": rng.normal(size=4).astype(np.float32),
            "vision_adapter.fc1.weight": rng.normal(size=(4, 4)).astype(np.float32),
            "vision_adapter.fc2.weight": rng.normal(size=(4, 4)).astype(np.float32),
            "vision_projection.weight": rng.normal(size=(6, 4)).astype(np.float32),
        }
        for name, value in weights.items():
            model.graph.initializers[name].const_value = ir.tensor(value)

        model_path = tmp_path / "muse-glimmer-parity.onnx"
        ir.save(model, str(model_path))
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        pixel_values = rng.normal(size=(4, 3)).astype(np.float32)
        actual = session.run(
            None,
            {
                "pixel_values": pixel_values,
                "image_grid_thw": np.array([[1, 2, 2]], dtype=np.int64),
            },
        )[0]

        hidden = pixel_values @ weights["vision_tower.patch_embedder.patch_embedding.weight"].T
        hidden += weights["vision_tower.patch_embedder.position_embedding_table.weight"]

        def layer_norm(values: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
            centered = values - np.mean(values, axis=-1, keepdims=True)
            variance = np.mean(np.square(centered), axis=-1, keepdims=True)
            return centered / np.sqrt(variance + vision.norm_eps) * weight + bias

        hidden = layer_norm(
            hidden,
            weights["vision_tower.ln_pre.weight"],
            weights["vision_tower.ln_pre.bias"],
        )
        hidden = layer_norm(
            hidden,
            weights["vision_tower.ln_post.weight"],
            weights["vision_tower.ln_post.bias"],
        )
        hidden = hidden @ weights["vision_adapter.fc1.weight"].T
        hidden = 0.5 * hidden * (1.0 + np.vectorize(math.erf)(hidden / np.sqrt(2.0)))
        hidden = hidden @ weights["vision_adapter.fc2.weight"].T
        hidden = 0.5 * hidden * (1.0 + np.vectorize(math.erf)(hidden / np.sqrt(2.0)))
        hidden = hidden @ weights["vision_projection.weight"].T
        expected = hidden / np.sqrt(
            np.mean(np.square(hidden), axis=-1, keepdims=True) + config.rms_norm_eps
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


class TestVlmRouting:
    """The text backbone decides which VLM builder assembles the pair."""

    @pytest.mark.parametrize(
        ("architecture", "expected"),
        [
            ("muse-glimmer", "build_muse_glimmer_vlm_from_gguf"),
            ("muse_glimmer", "build_muse_glimmer_vlm_from_gguf"),
            ("gemma4", "build_gemma4_vlm_from_gguf"),
        ],
    )
    def test_routes_on_the_text_architecture(
        self, tmp_path: Path, architecture: str, expected: str
    ):
        from mobius.integrations.gguf._mmproj import build_vlm_from_gguf

        text_path = tmp_path / "text.gguf"
        _write_minimal_gguf(text_path, architecture)
        mmproj_path = tmp_path / "mmproj.gguf"
        if expected == "build_gemma4_vlm_from_gguf":
            _write_clip_mmproj_gguf(mmproj_path)
        else:
            _write_muse_glimmer_mmproj_gguf(mmproj_path)
        package = mock.sentinel.package

        with mock.patch(
            f"mobius.integrations.gguf._mmproj.{expected}", return_value=package
        ) as builder:
            actual = build_vlm_from_gguf(text_path, mmproj_path, keep_quantized=False)

        assert actual is package
        builder.assert_called_once_with(
            str(text_path),
            str(mmproj_path),
            dtype=None,
            execution_provider="default",
            keep_quantized=False,
            _text_gguf_model=mock.ANY,
            _mmproj_gguf_model=mock.ANY,
        )

    @pytest.mark.parametrize(
        ("architecture", "expected"),
        [
            ("gemma4", "build_gemma4_vlm_from_gguf"),
            ("muse-glimmer", "build_muse_glimmer_vlm_from_gguf"),
        ],
    )
    def test_remote_clip_companion_preflight_reaches_vlm_builder(
        self, tmp_path: Path, architecture: str, expected: str
    ) -> None:
        from mobius.integrations.gguf._mmproj import build_vlm_from_gguf

        text_path = tmp_path / "text.gguf"
        _write_minimal_gguf(text_path, architecture)
        mmproj_path = tmp_path / "downloaded-mmproj.gguf"
        if expected == "build_gemma4_vlm_from_gguf":
            _write_clip_mmproj_gguf(mmproj_path)
        else:
            _write_muse_glimmer_mmproj_gguf(mmproj_path)
        remote_ref = f"example/{architecture}:mmproj.gguf"
        package = mock.sentinel.package
        resolved_revision = "a" * 40

        with (
            mock.patch("mobius.integrations.gguf._builder.HfApi") as api_type,
            mock.patch(
                "mobius.integrations.gguf._builder._preflight_hf_mmproj_companion_file",
                return_value=resolved_revision,
            ) as selected_file_preflight,
            mock.patch(
                "mobius.integrations.gguf._builder.hf_hub_download",
                return_value=str(mmproj_path),
            ) as download,
            mock.patch(
                f"mobius.integrations.gguf._mmproj.{expected}",
                return_value=package,
            ) as builder,
        ):
            api_type.return_value.model_info.return_value = SimpleNamespace(
                gguf={"architecture": architecture}
            )
            actual = build_vlm_from_gguf(text_path, remote_ref, keep_quantized=False)

        assert actual is package
        api_type.return_value.model_info.assert_not_called()
        selected_file_preflight.assert_called_once_with(
            f"example/{architecture}",
            "mmproj.gguf",
            revision="main",
        )
        download.assert_called_once_with(
            repo_id=f"example/{architecture}",
            filename="mmproj.gguf",
            revision=resolved_revision,
        )
        builder.assert_called_once_with(
            str(text_path),
            str(mmproj_path),
            dtype=None,
            execution_provider="default",
            keep_quantized=False,
            _text_gguf_model=mock.ANY,
            _mmproj_gguf_model=mock.ANY,
        )

    def test_remote_non_clip_companion_rejects_before_download_or_builder(
        self, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf._mmproj import build_vlm_from_gguf

        text_path = tmp_path / "text.gguf"
        _write_minimal_gguf(text_path, "gemma4")
        remote_ref = "example/wrong-companion:model.gguf"

        with (
            mock.patch("mobius.integrations.gguf._builder.HfApi") as api_type,
            mock.patch(
                "mobius.integrations.gguf._builder._preflight_hf_mmproj_companion_file",
                side_effect=ValueError(
                    "Expected a 'clip' mmproj GGUF, got architecture 'llama'."
                ),
            ) as selected_file_preflight,
            mock.patch("mobius.integrations.gguf._builder.hf_hub_download") as download,
            mock.patch(
                "mobius.integrations.gguf._mmproj.build_gemma4_vlm_from_gguf"
            ) as builder,
            pytest.raises(ValueError, match=r"Expected a 'clip' mmproj.*architecture 'llama'"),
        ):
            api_type.return_value.model_info.return_value = SimpleNamespace(
                gguf={"architecture": "clip"}
            )
            build_vlm_from_gguf(text_path, remote_ref)

        api_type.return_value.model_info.assert_not_called()
        selected_file_preflight.assert_called_once_with(
            "example/wrong-companion",
            "model.gguf",
            revision="main",
        )
        download.assert_not_called()
        builder.assert_not_called()

    def test_downloaded_mmproj_header_is_revalidated_before_builder(
        self, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf._mmproj import build_vlm_from_gguf

        text_path = tmp_path / "text.gguf"
        downloaded_path = tmp_path / "selected-file.gguf"
        _write_minimal_gguf(text_path, "gemma4")
        _write_minimal_gguf(downloaded_path, "llama")
        remote_ref = "example/toctou:mmproj.gguf"

        with (
            mock.patch(
                "mobius.integrations.gguf._builder._preflight_hf_mmproj_companion_file",
                return_value="b" * 40,
            ),
            mock.patch(
                "mobius.integrations.gguf._builder.hf_hub_download",
                return_value=str(downloaded_path),
            ),
            mock.patch(
                "mobius.integrations.gguf._mmproj.build_gemma4_vlm_from_gguf"
            ) as builder,
            pytest.raises(ValueError, match=r"Expected a 'clip' mmproj.*architecture 'llama'"),
        ):
            build_vlm_from_gguf(text_path, remote_ref)

        builder.assert_not_called()


class TestMuseGlimmerTemporalPatchSize:
    """The temporal depth is read off the weight, not assumed."""

    def test_conv2d_kernel_means_a_single_frame(self) -> None:
        from mobius.integrations.gguf._mmproj import _muse_glimmer_temporal_patch_size

        weight = np.zeros((1536, 3, 14, 14), dtype=np.float32)
        assert _muse_glimmer_temporal_patch_size(weight, patch_size=14) == 1

    def test_conv3d_kernel_keeps_its_temporal_depth(self) -> None:
        from mobius.integrations.gguf._mmproj import _muse_glimmer_temporal_patch_size

        weight = np.zeros((1536, 3, 2, 14, 14), dtype=np.float32)
        assert _muse_glimmer_temporal_patch_size(weight, patch_size=14) == 2

    def test_indivisible_kernel_is_rejected(self) -> None:
        from mobius.integrations.gguf._mmproj import _muse_glimmer_temporal_patch_size

        weight = np.zeros((1536, 3, 13, 14), dtype=np.float32)
        with pytest.raises(ValueError, match="not divisible"):
            _muse_glimmer_temporal_patch_size(weight, patch_size=14)


class TestMuseGlimmerMediaTokenIds:
    """A GGUF carries no tokenizer, so the media placeholder ids arrive unset.

    Leaving them unset is not cosmetic. The embedding graph compares
    ``input_ids`` against them, and the ort-genai exporter drops the field from
    ``genai_config`` when it is ``None``, so the exported package loses the
    ability to address video entirely.
    """

    @staticmethod
    def _config(**overrides):
        from mobius._configs import MuseGlimmerConfig

        values = dict(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
            vocab_size=256,
        )
        values.update(overrides)
        return MuseGlimmerConfig(**values)

    def test_fills_in_the_published_ids(self) -> None:
        from mobius.integrations.gguf._mmproj import _with_muse_glimmer_media_token_ids
        from mobius.models.muse_glimmer import IMAGE_TOKEN_ID, VIDEO_TOKEN_ID

        config = self._config()
        assert config.image_token_id is None
        assert config.video_token_id is None

        result = _with_muse_glimmer_media_token_ids(config)

        assert result.image_token_id == IMAGE_TOKEN_ID
        assert result.video_token_id == VIDEO_TOKEN_ID

    def test_does_not_override_ids_the_caller_supplied(self) -> None:
        from mobius.integrations.gguf._mmproj import _with_muse_glimmer_media_token_ids

        result = _with_muse_glimmer_media_token_ids(
            self._config(image_token_id=7, video_token_id=9)
        )

        assert result.image_token_id == 7
        assert result.video_token_id == 9

    def test_the_embedding_graph_never_compares_against_none(self) -> None:
        from mobius.models.muse_glimmer import (
            VIDEO_TOKEN_ID,
            MuseGlimmerEmbeddingModel,
        )

        module = MuseGlimmerEmbeddingModel(self._config())

        assert module._video_token_id == VIDEO_TOKEN_ID
