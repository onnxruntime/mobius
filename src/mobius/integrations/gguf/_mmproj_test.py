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


def test_generic_mlp_norm_variant_fails_closed():
    from mobius._configs._sub_configs import VisionConfig
    from mobius.integrations.gguf._mmproj import _generic_projector_dimensions

    shapes = {
        "mm.0.weight": (16, 8),
        "mm.2.weight": (16, 16),
    }
    sidecar = SimpleNamespace(
        tensor_names=[*shapes, "mm.3.weight"],
        get_tensor_shape=shapes.__getitem__,
    )
    vision = VisionConfig(image_size=28, patch_size=14, hidden_size=8)

    with pytest.raises(ValueError, match="MLP_NORM"):
        _generic_projector_dimensions(sidecar, "mlp", vision)


def test_generic_projector_unknown_variant_fails_closed():
    from mobius._configs._sub_configs import VisionConfig
    from mobius.integrations.gguf._mmproj import _generic_projector_dimensions

    vision = VisionConfig(image_size=28, patch_size=14, hidden_size=8)
    sidecar = SimpleNamespace(get_tensor_shape=lambda _: ())

    with pytest.raises(ValueError, match="Unknown generic"):
        _generic_projector_dimensions(sidecar, "future-projector", vision)


def test_text_gguf_opener_preserves_resolved_shard_manifest(monkeypatch):
    from mobius.integrations.gguf import _mmproj, _shard_set

    resolved = SimpleNamespace(
        shard_paths=["a.gguf", "b.gguf"],
        expected_sha256={"a.gguf": "a" * 64, "b.gguf": "b" * 64},
        expected_sizes={"a.gguf": 1, "b.gguf": 2},
    )
    opened = object()
    open_model = mock.Mock(return_value=opened)
    monkeypatch.setattr(_shard_set, "open_gguf_model", open_model)

    assert _mmproj._open_text_gguf(resolved) is opened
    open_model.assert_called_once_with(resolved)


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


def _write_gemma3_mmproj_gguf(
    path: Path,
    *,
    extra_tensor: str | None = None,
) -> None:
    """Write the exact 1-block Gemma3 closure in llama.cpp orientation."""
    from gguf import GGUFWriter

    hidden = _VISION_HIDDEN
    intermediate = _VISION_FFN
    writer = GGUFWriter(str(path), "clip")
    writer.add_string("clip.projector_type", "gemma3")
    writer.add_bool("clip.has_vision_encoder", True)
    writer.add_uint32("clip.vision.embedding_length", hidden)
    writer.add_uint32("clip.vision.feed_forward_length", intermediate)
    writer.add_uint32("clip.vision.block_count", 1)
    writer.add_uint32("clip.vision.attention.head_count", _VISION_HEADS)
    writer.add_uint32("clip.vision.image_size", _IMAGE_SIZE)
    writer.add_uint32("clip.vision.patch_size", _PATCH_SIZE)
    writer.add_uint32("clip.vision.projection_dim", _TEXT_HIDDEN)
    writer.add_array("clip.vision.image_mean", [0.5, 0.5, 0.5])
    writer.add_array("clip.vision.image_std", [0.5, 0.5, 0.5])
    writer.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-6)

    def add(name: str, shape: tuple[int, ...]) -> None:
        writer.add_tensor(name, np.random.randn(*shape).astype(np.float32))

    add("v.patch_embd.weight", (hidden, 3, _PATCH_SIZE, _PATCH_SIZE))
    add("v.patch_embd.bias", (hidden,))
    add("v.position_embd.weight", ((_IMAGE_SIZE // _PATCH_SIZE) ** 2, hidden))
    add("v.post_ln.weight", (hidden,))
    add("v.post_ln.bias", (hidden,))
    add("mm.soft_emb_norm.weight", (hidden,))
    add("mm.input_projection.weight", (hidden, _TEXT_HIDDEN))
    prefix = "v.blk.0."
    for stem in ("ln1", "ln2"):
        add(prefix + stem + ".weight", (hidden,))
        add(prefix + stem + ".bias", (hidden,))
    for stem in ("attn_q", "attn_k", "attn_v", "attn_out"):
        add(prefix + stem + ".weight", (hidden, hidden))
        add(prefix + stem + ".bias", (hidden,))
    add(prefix + "ffn_down.weight", (intermediate, hidden))
    add(prefix + "ffn_down.bias", (intermediate,))
    add(prefix + "ffn_up.weight", (hidden, intermediate))
    add(prefix + "ffn_up.bias", (hidden,))
    if extra_tensor is not None:
        add(extra_tensor, (1,))

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


def _write_quantized_gemma4_text_gguf(path: Path, *, float_projection: bool = False) -> None:
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
        if float_projection and layer == 0:
            _f32(f"{prefix}.attn_q.weight", (num_heads * head_dim, hidden_size))
        else:
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


def _write_qwen_vl_pair(
    text_path: Path,
    mmproj_path: Path,
    *,
    projector_type: str,
) -> None:
    """Write a tiny exact Qwen2/Qwen2.5-VL text + sidecar pair."""
    from gguf import GGUFWriter

    text_hidden = 16
    vision_hidden = 8
    vision_intermediate = 12
    layers = 2
    heads = 2
    vocab = 32
    tokens = [f"token-{index}" for index in range(vocab)]
    tokens[1:5] = [
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]

    text = GGUFWriter(str(text_path), "qwen2vl")
    text.add_string("general.name", "Tiny Qwen VL")
    text.add_context_length(64)
    text.add_embedding_length(text_hidden)
    text.add_feed_forward_length(32)
    text.add_block_count(2)
    text.add_head_count(2)
    text.add_head_count_kv(2)
    text.add_rope_freq_base(1_000_000.0)
    text.add_layer_norm_rms_eps(1e-6)
    text.add_array("qwen2vl.rope.dimension_sections", [1, 1, 2, 0])
    text.add_tokenizer_model("gpt2")
    text.add_string("tokenizer.ggml.pre", "qwen2")
    text.add_token_list(tokens)
    text.add_token_merges(["token-5 token-6"])

    rng = np.random.default_rng(17)

    def _text_tensor(name: str, shape: tuple[int, ...]) -> None:
        text.add_tensor(name, rng.normal(size=shape).astype(np.float32))

    _text_tensor("token_embd.weight", (vocab, text_hidden))
    _text_tensor("output_norm.weight", (text_hidden,))
    for layer in range(2):
        prefix = f"blk.{layer}."
        _text_tensor(prefix + "attn_norm.weight", (text_hidden,))
        _text_tensor(prefix + "ffn_norm.weight", (text_hidden,))
        for stem in ("attn_q", "attn_k", "attn_v"):
            _text_tensor(prefix + stem + ".weight", (text_hidden, text_hidden))
            _text_tensor(prefix + stem + ".bias", (text_hidden,))
        _text_tensor(prefix + "attn_output.weight", (text_hidden, text_hidden))
        for stem in ("ffn_gate", "ffn_up"):
            _text_tensor(prefix + stem + ".weight", (32, text_hidden))
        _text_tensor(prefix + "ffn_down.weight", (text_hidden, 32))
    text.write_header_to_file()
    text.write_kv_data_to_file()
    text.write_tensors_to_file()
    text.close()

    sidecar = GGUFWriter(str(mmproj_path), "clip")
    sidecar.add_string("general.name", "Tiny Qwen VL")
    sidecar.add_string("general.type", "clip-vision")
    sidecar.add_bool("clip.has_vision_encoder", True)
    sidecar.add_string("clip.projector_type", projector_type)
    sidecar.add_uint32("clip.vision.embedding_length", vision_hidden)
    sidecar.add_uint32(
        "clip.vision.feed_forward_length",
        text_hidden if projector_type == "qwen2vl_merger" else vision_intermediate,
    )
    sidecar.add_uint32("clip.vision.block_count", layers)
    sidecar.add_uint32("clip.vision.attention.head_count", heads)
    sidecar.add_uint32("clip.vision.image_size", 8)
    sidecar.add_uint32("clip.vision.patch_size", 2)
    sidecar.add_uint32("clip.vision.projection_dim", text_hidden)
    sidecar.add_array("clip.vision.image_mean", [0.48145466, 0.4578275, 0.40821073])
    sidecar.add_array("clip.vision.image_std", [0.26862954, 0.2613026, 0.2757771])
    sidecar.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-6)
    if projector_type == "qwen2.5vl_merger":
        sidecar.add_bool("clip.use_silu", True)
        sidecar.add_uint32("clip.vision.n_wa_pattern", 2)

    sequence = 1.0

    def _sidecar_tensor(name: str, shape: tuple[int, ...]) -> None:
        nonlocal sequence
        count = int(np.prod(shape))
        values = np.arange(sequence, sequence + count, dtype=np.float32).reshape(shape)
        sequence += count
        sidecar.add_tensor(name, values)

    for name in ("v.patch_embd.weight", "v.patch_embd.weight.1"):
        _sidecar_tensor(name, (vision_hidden, 3, 2, 2))
    for layer in range(layers):
        prefix = f"v.blk.{layer}."
        for stem in ("attn_q", "attn_k", "attn_v", "attn_out"):
            _sidecar_tensor(prefix + stem + ".weight", (vision_hidden, vision_hidden))
            _sidecar_tensor(prefix + stem + ".bias", (vision_hidden,))
        if projector_type == "qwen2vl_merger":
            for stem in ("ln1", "ln2"):
                _sidecar_tensor(prefix + stem + ".weight", (vision_hidden,))
                _sidecar_tensor(prefix + stem + ".bias", (vision_hidden,))
            _sidecar_tensor(prefix + "ffn_up.weight", (vision_hidden, vision_intermediate))
            _sidecar_tensor(prefix + "ffn_up.bias", (vision_hidden,))
            _sidecar_tensor(prefix + "ffn_down.weight", (vision_intermediate, vision_hidden))
            _sidecar_tensor(prefix + "ffn_down.bias", (vision_intermediate,))
        else:
            for stem in ("ln1", "ln2"):
                _sidecar_tensor(prefix + stem + ".weight", (vision_hidden,))
            for stem in ("ffn_gate", "ffn_up"):
                _sidecar_tensor(
                    prefix + stem + ".weight", (vision_intermediate, vision_hidden)
                )
                _sidecar_tensor(prefix + stem + ".bias", (vision_intermediate,))
            _sidecar_tensor(prefix + "ffn_down.weight", (vision_hidden, vision_intermediate))
            _sidecar_tensor(prefix + "ffn_down.bias", (vision_hidden,))
    _sidecar_tensor("v.post_ln.weight", (vision_hidden,))
    if projector_type == "qwen2vl_merger":
        _sidecar_tensor("v.post_ln.bias", (vision_hidden,))
    merged = vision_hidden * 4
    _sidecar_tensor("mm.0.weight", (merged, merged))
    _sidecar_tensor("mm.0.bias", (merged,))
    _sidecar_tensor("mm.2.weight", (text_hidden, merged))
    _sidecar_tensor("mm.2.bias", (text_hidden,))
    sidecar.write_header_to_file()
    sidecar.write_kv_data_to_file()
    sidecar.write_tensors_to_file()
    sidecar.close()


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


class TestGemma3Preflight:
    def _pair(self, tmp_path: Path, *, extra_tensor: str | None = None):
        from mobius.integrations.gguf._reader import GGUFModel

        text_path = tmp_path / "gemma3.gguf"
        mmproj_path = tmp_path / "mmproj.gguf"
        _write_minimal_gguf(text_path, "gemma3")
        _write_gemma3_mmproj_gguf(mmproj_path, extra_tensor=extra_tensor)
        return GGUFModel(str(text_path)), GGUFModel(str(mmproj_path))

    def test_exact_tensor_closure_accepts_pinned_legacy_identity(self, tmp_path: Path):
        from mobius.integrations.gguf._mmproj import _preflight_mmproj_pair
        from mobius.integrations.gguf._mmproj_registry import MMProjModality

        text, mmproj = self._pair(tmp_path)
        specs = _preflight_mmproj_pair(text, mmproj, modalities=(MMProjModality.VISION,))
        assert specs[MMProjModality.VISION].projector_type == "gemma3"
        assert len(mmproj.tensor_names) == 23

    def test_generic_dispatch_resolves_only_the_exact_gemma3_pair(self):
        from mobius.integrations.gguf._mmproj import (
            _resolve_vlm_builder,
            build_gemma3_vlm_from_gguf,
        )

        assert _resolve_vlm_builder("gemma3", "gemma3") is build_gemma3_vlm_from_gguf
        with pytest.raises(ValueError, match="targets"):
            _resolve_vlm_builder("gemma4", "gemma3")

    def test_exact_tensor_closure_rejects_unknown_tensor(self, tmp_path: Path):
        from mobius.integrations.gguf._mmproj import _preflight_mmproj_pair
        from mobius.integrations.gguf._mmproj_registry import MMProjModality

        text, mmproj = self._pair(tmp_path, extra_tensor="v.pre_ln.weight")
        with pytest.raises(ValueError, match=r"outside the pinned.*closure"):
            _preflight_mmproj_pair(text, mmproj, modalities=(MMProjModality.VISION,))

    @pytest.mark.parametrize("qtype", ["F32", "F16", "BF16"])
    def test_float_storage_types_are_accepted(self, tmp_path: Path, qtype: str):
        from mobius.integrations.gguf._mmproj import _preflight_mmproj_pair
        from mobius.integrations.gguf._mmproj_registry import MMProjModality

        text, mmproj = self._pair(tmp_path)
        with mock.patch.object(
            mmproj,
            "get_tensor_type",
            return_value=SimpleNamespace(name=qtype),
        ):
            _preflight_mmproj_pair(text, mmproj, modalities=(MMProjModality.VISION,))

    def test_packed_vision_tensor_is_rejected(self, tmp_path: Path):
        from mobius.integrations.gguf._mmproj import _preflight_mmproj_pair
        from mobius.integrations.gguf._mmproj_registry import MMProjModality

        text, mmproj = self._pair(tmp_path)
        original = mmproj.get_tensor_type
        with (
            mock.patch.object(
                mmproj,
                "get_tensor_type",
                side_effect=lambda name: (
                    SimpleNamespace(name="Q4_K")
                    if name == "v.patch_embd.weight"
                    else original(name)
                ),
            ),
            pytest.raises(NotImplementedError, match="packed Q4_K"),
        ):
            _preflight_mmproj_pair(text, mmproj, modalities=(MMProjModality.VISION,))

    def test_config_derives_4x4_pool_and_soft_tokens(self, tmp_path: Path):
        from mobius.integrations.gguf._mmproj import (
            read_mmproj_gemma3_vision_config,
        )

        _, mmproj = self._pair(tmp_path)
        config = read_mmproj_gemma3_vision_config(mmproj)
        assert config.pooling_kernel_size == 4
        assert config.mm_tokens_per_image == 1
        assert config.position_embedding_size == 16

    def test_values_map_without_transpose_and_projector_norm_is_unoffset(self, tmp_path: Path):
        from mobius.integrations.gguf._mmproj import (
            _mmproj_gemma3_vision_to_hf,
        )

        _, mmproj = self._pair(tmp_path)
        mapped = _mmproj_gemma3_vision_to_hf(mmproj)
        np.testing.assert_array_equal(
            mapped["vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight"].numpy(),
            mmproj.get_tensor("v.blk.0.ffn_down.weight"),
        )
        np.testing.assert_allclose(
            mapped["multi_modal_projector.mm_soft_emb_norm.weight"].numpy(),
            mmproj.get_tensor("mm.soft_emb_norm.weight") - 1.0,
        )


class TestQwenVLMMProj:
    @pytest.mark.parametrize(
        "projector_type,expected_intermediate,expected_full_attention",
        [
            ("qwen2vl_merger", 12, None),
            ("qwen2.5vl_merger", 12, [1]),
        ],
    )
    def test_real_contract_config_and_exact_closure(
        self,
        tmp_path: Path,
        projector_type: str,
        expected_intermediate: int,
        expected_full_attention: list[int] | None,
    ):
        from mobius.integrations.gguf._mmproj import (
            _preflight_mmproj_pair,
            read_mmproj_qwen_vision_config,
        )
        from mobius.integrations.gguf._mmproj_registry import MMProjModality
        from mobius.integrations.gguf._reader import GGUFModel

        text_path = tmp_path / "text.gguf"
        sidecar_path = tmp_path / "mmproj.gguf"
        _write_qwen_vl_pair(
            text_path,
            sidecar_path,
            projector_type=projector_type,
        )
        text = GGUFModel(str(text_path))
        sidecar = GGUFModel(str(sidecar_path))

        spec = _preflight_mmproj_pair(
            text,
            sidecar,
            modalities=(MMProjModality.VISION,),
        )[MMProjModality.VISION]
        vision = read_mmproj_qwen_vision_config(sidecar, spec.projector_type)

        assert vision is not None
        assert vision.intermediate_size == expected_intermediate
        assert vision.fullatt_block_indexes == expected_full_attention
        assert vision.temporal_patch_size == 2
        assert vision.spatial_merge_size == 2
        assert vision.out_hidden_size == 16

    @pytest.mark.parametrize("projector_type", ["qwen2vl_merger", "qwen2.5vl_merger"])
    def test_qwen_tensor_transform_values(
        self,
        tmp_path: Path,
        projector_type: str,
    ):
        from mobius.integrations.gguf._mmproj import _mmproj_qwen_vision_to_hf
        from mobius.integrations.gguf._reader import GGUFModel

        text_path = tmp_path / "text.gguf"
        sidecar_path = tmp_path / "mmproj.gguf"
        _write_qwen_vl_pair(
            text_path,
            sidecar_path,
            projector_type=projector_type,
        )
        sidecar = GGUFModel(str(sidecar_path))
        state = _mmproj_qwen_vision_to_hf(sidecar, projector_type)

        patch0 = np.array(sidecar.get_tensor("v.patch_embd.weight"))
        patch1 = np.array(sidecar.get_tensor("v.patch_embd.weight.1"))
        np.testing.assert_array_equal(
            state["visual.patch_embed.proj.weight"].numpy(),
            np.stack([patch0, patch1], axis=2),
        )
        qkv = np.concatenate(
            [
                np.array(sidecar.get_tensor(f"v.blk.0.attn_{stem}.weight"))
                for stem in ("q", "k", "v")
            ],
            axis=0,
        )
        np.testing.assert_array_equal(
            state["visual.blocks.0.attn.qkv.weight"].numpy(),
            qkv,
        )
        if projector_type == "qwen2vl_merger":
            np.testing.assert_array_equal(
                state["visual.blocks.0.mlp.down_proj.weight"].numpy(),
                np.array(sidecar.get_tensor("v.blk.0.ffn_up.weight")),
            )
            np.testing.assert_array_equal(
                state["visual.blocks.0.mlp.up_proj.weight"].numpy(),
                np.array(sidecar.get_tensor("v.blk.0.ffn_down.weight")),
            )

    @pytest.mark.parametrize("projector_type", ["qwen2vl_merger", "qwen2.5vl_merger"])
    def test_builds_canonical_package_and_runs_mixed_media(
        self,
        tmp_path: Path,
        projector_type: str,
    ):
        import onnx_ir as ir
        import onnxruntime as ort

        from mobius.integrations.gguf._mmproj import build_qwen_vlm_from_gguf

        text_path = tmp_path / "text.gguf"
        sidecar_path = tmp_path / "mmproj.gguf"
        _write_qwen_vl_pair(
            text_path,
            sidecar_path,
            projector_type=projector_type,
        )
        package = build_qwen_vlm_from_gguf(
            text_path,
            sidecar_path,
            keep_quantized=False,
        )

        assert set(package) == {"decoder", "vision_encoder", "embedding"}
        assert [value.name for value in package["vision_encoder"].graph.inputs] == [
            "pixel_values",
            "image_grid_thw",
        ]
        assert package["vision_encoder"].graph.inputs[0].dtype == ir.DataType.FLOAT
        assert [value.name for value in package["embedding"].graph.inputs] == [
            "input_ids",
            "image_features",
            "video_features",
        ]
        assert package["decoder"].graph.inputs[0].name == "inputs_embeds"
        assert package["decoder"].graph.inputs[2].shape[0] == 3

        vision_session = ort.InferenceSession(
            ir.serde.serialize_model(package["vision_encoder"]).SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        vision_result = vision_session.run(
            None,
            {
                "pixel_values": np.arange(96, dtype=np.float32).reshape(4, 24) / 100,
                "image_grid_thw": np.array([[1, 2, 2]], dtype=np.int64),
            },
        )[0]
        assert vision_result.shape == (1, 16)
        assert np.isfinite(vision_result).all()

        embedding_path = tmp_path / "embedding.onnx"
        ir.save(package["embedding"], str(embedding_path))
        session = ort.InferenceSession(
            str(embedding_path),
            providers=["CPUExecutionProvider"],
        )
        input_ids = np.array([[3, 4, 0], [4, 3, 0]], dtype=np.int64)
        images = np.array([[10.0] * 16, [20.0] * 16], dtype=np.float32)
        videos = np.array([[30.0] * 16, [40.0] * 16], dtype=np.float32)
        result = session.run(
            None,
            {
                "input_ids": input_ids,
                "image_features": images,
                "video_features": videos,
            },
        )[0]
        np.testing.assert_array_equal(result[0, 0], images[0])
        np.testing.assert_array_equal(result[1, 1], images[1])
        np.testing.assert_array_equal(result[0, 1], videos[0])
        np.testing.assert_array_equal(result[1, 0], videos[1])

        text_only = session.run(
            None,
            {
                "input_ids": np.zeros((2, 1), dtype=np.int64),
                "image_features": np.empty((0, 16), dtype=np.float32),
                "video_features": np.empty((0, 16), dtype=np.float32),
            },
        )[0]
        assert text_only.shape == (2, 1, 16)
        assert np.isfinite(text_only).all()

        decoder_session = ort.InferenceSession(
            ir.serde.serialize_model(package["decoder"]).SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        prefill_feeds = {
            "inputs_embeds": result[:1, :2],
            "attention_mask": np.ones((1, 2), dtype=np.int64),
            "position_ids": np.zeros((3, 1, 2), dtype=np.int64),
        }
        for layer in range(2):
            for cache_type in ("key", "value"):
                prefill_feeds[f"past_key_values.{layer}.{cache_type}"] = np.empty(
                    (1, 2, 0, 8),
                    dtype=np.float32,
                )
        prefill = decoder_session.run(None, prefill_feeds)
        assert prefill[0].shape == (1, 2, 32)

        decode_feeds = {
            "inputs_embeds": text_only[:1],
            "attention_mask": np.ones((1, 3), dtype=np.int64),
            "position_ids": np.full((3, 1, 1), 2, dtype=np.int64),
        }
        for layer in range(2):
            decode_feeds[f"past_key_values.{layer}.key"] = prefill[1 + 2 * layer]
            decode_feeds[f"past_key_values.{layer}.value"] = prefill[2 + 2 * layer]
        decode = decoder_session.run(None, decode_feeds)
        assert decode[0].shape == (1, 1, 32)
        assert all(cache.shape[2] == 3 for cache in decode[1:])


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
        assert package.gguf_quantization_report.storage_quantized is True
        # The float mmproj vision encoder is merged into the package-level
        # report alongside the quantized text backbone (issue 4): the target
        # storage format spans both.
        assert package.gguf_quantization_report.target_storage_format == (
            "INT4 affine block-32 + float"
        )

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
        assert reloaded.gguf_quantization_report == package.gguf_quantization_report
        assert set(reloaded) == {"decoder", "vision_encoder", "embedding"}
        assert all(
            initializer.const_value is not None
            for model in reloaded.values()
            for initializer in model.graph.initializers.values()
        )

    def test_float_projection_in_quantized_text_fails_closed(
        self, clip_mmproj_gguf: Path, tmp_path: Path
    ):
        from mobius.integrations.gguf import build_gemma4_vlm_from_gguf

        text_gguf = tmp_path / "gemma4-q4-f32-projection.gguf"
        _write_quantized_gemma4_text_gguf(text_gguf, float_projection=True)

        with pytest.raises(ValueError, match=r"would quantize a source-float tensor"):
            build_gemma4_vlm_from_gguf(text_gguf, clip_mmproj_gguf, image_token_id=63)

        package = build_gemma4_vlm_from_gguf(
            text_gguf,
            clip_mmproj_gguf,
            image_token_id=63,
            keep_quantized=False,
        )
        assert "MatMulNBits" not in _component_op_types(package["decoder"])

    @pytest.mark.parametrize(
        "tensor_name",
        ["per_layer_token_embd.weight", "per_layer_model_proj.weight"],
    )
    def test_quantized_per_layer_table_fails_closed(self, tensor_name: str):
        from types import SimpleNamespace

        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf._mmproj import (
            _text_gguf_to_hf_multimodal_quantized,
        )

        class _PackedPerLayerModel:
            def tensor_items_raw(self):
                yield (
                    tensor_name,
                    np.empty(0, dtype=np.uint8),
                    GGMLQuantizationType.Q4_0,
                    (2, 32, 32),
                )

        config = SimpleNamespace(
            quantization=SimpleNamespace(
                quantize_embeddings=False,
                quantize_lm_head=False,
            ),
            tie_word_embeddings=False,
        )
        with pytest.raises(ValueError, match=r"cannot retain packed tensor"):
            _text_gguf_to_hf_multimodal_quantized(
                _PackedPerLayerModel(),
                config,
                bits=4,
                block_size=32,
                symmetric=True,
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


class TestMmprojQuantizationReportPreflight:
    """Unit tests for the mmproj vision/(audio) preflight (follow-up issue 4).

    ``build_gemma4_vlm_from_gguf`` previously computed
    ``pkg.gguf_quantization_report`` from the text GGUF alone, silently
    excluding every mmproj vision/(audio) source tensor from the census and
    tensor records. These tests exercise
    ``_preflight_mmproj_quantization_report`` directly against a synthetic
    ``clip`` mmproj GGUF.
    """

    def test_vision_only_census_covers_whole_file_and_records_are_mapped_vision(
        self, tmp_path: Path
    ):
        from mobius.integrations.gguf._mmproj import (
            _preflight_mmproj_quantization_report,
        )
        from mobius.integrations.gguf._mmproj_mapping import map_mmproj_vision_to_hf
        from mobius.integrations.gguf._quantization_report import QuantizationDisposition
        from mobius.integrations.gguf._reader import GGUFModel

        path = tmp_path / "mmproj.gguf"
        _write_clip_mmproj_gguf(path, with_audio=True)
        mmproj_gguf = GGUFModel(str(path))

        report = _preflight_mmproj_quantization_report(mmproj_gguf, include_audio=False)

        # The source qtype census covers *every* tensor in the file (vision +
        # audio + projector), independent of whether audio is mapped below.
        assert sum(stat.tensor_count for stat in report.source_qtype_census) == (
            mmproj_gguf.num_tensors
        )
        assert sum(stat.source_bytes for stat in report.source_qtype_census) == sum(
            int(tensor.n_bytes) for tensor in mmproj_gguf.reader_tensors()
        )
        # This fixture is entirely float32.
        assert [stat.qtype for stat in report.source_qtype_census] == ["F32"]

        # Only vision (+ shared projector) tensors are *mapped* into records
        # when include_audio=False, even though audio tensors are present in
        # the file and already counted in the census above.
        expected_names = {
            f"mmproj:{name}"
            for name in mmproj_gguf.tensor_names
            if (name.startswith("v.") or name == "mm.input_projection.weight")
            and map_mmproj_vision_to_hf(name) is not None
        }
        assert expected_names  # sanity: the fixture does map some tensors
        assert {record.name for record in report.tensor_records} == expected_names
        assert not any(record.name.startswith("mmproj:a.") for record in report.tensor_records)
        assert all(
            record.disposition is QuantizationDisposition.SOURCE_FLOAT
            for record in report.tensor_records
        )
        assert report.explicit_float_tensors == report.tensor_records
        assert report.source_fidelity is True
        assert report.storage_quantized is False
        assert report.target_storage_format == "float"

    def test_include_audio_adds_audio_records_without_changing_the_file_census(
        self, tmp_path: Path
    ):
        from mobius.integrations.gguf._mmproj import (
            _preflight_mmproj_quantization_report,
        )
        from mobius.integrations.gguf._mmproj_mapping import (
            map_mmproj_audio_to_hf,
        )
        from mobius.integrations.gguf._reader import GGUFModel

        path = tmp_path / "mmproj.gguf"
        _write_clip_mmproj_gguf(path, with_audio=True)
        mmproj_gguf = GGUFModel(str(path))

        vision_only = _preflight_mmproj_quantization_report(mmproj_gguf, include_audio=False)
        vision_and_audio = _preflight_mmproj_quantization_report(
            mmproj_gguf, include_audio=True
        )

        # Enabling audio only adds mapped records; the raw file census (every
        # tensor's qtype/bytes) is a property of the file, not of what gets
        # mapped, so it must be unaffected by include_audio.
        assert vision_and_audio.source_qtype_census == vision_only.source_qtype_census
        assert len(vision_and_audio.tensor_records) > len(vision_only.tensor_records)

        expected_audio_names = {
            f"mmproj:{name}"
            for name in mmproj_gguf.tensor_names
            if (name.startswith("a.") or name == "mm.a.input_projection.weight")
            and map_mmproj_audio_to_hf(name) is not None
        }
        assert expected_audio_names  # the with_audio fixture maps audio tensors
        record_names = {record.name for record in vision_and_audio.tensor_records}
        assert expected_audio_names <= record_names
        vision_names = {record.name for record in vision_only.tensor_records}
        assert expected_audio_names.isdisjoint(vision_names)
        assert vision_names <= record_names

    def test_quantized_mmproj_tensor_dequantizes_to_float_and_breaks_fidelity(
        self, tmp_path: Path
    ):
        """Report atypical quantized mmproj tensors as dequantized float.

        The encoder always builds a float parameter, so the report must show
        dequantization rather than silently claiming perfect source fidelity.
        """
        from gguf import GGMLQuantizationType, GGUFWriter

        from mobius.integrations.gguf._mmproj import (
            _preflight_mmproj_quantization_report,
        )
        from mobius.integrations.gguf._quantization_report import QuantizationDisposition
        from mobius.integrations.gguf._reader import GGUFModel

        path = tmp_path / "mmproj-quantized-patch.gguf"
        writer = GGUFWriter(str(path), "clip")
        writer.add_string("clip.vision.projector_type", "gemma4v")
        writer.add_bool("clip.has_vision_encoder", True)
        # A Q8_0 block is 32 elements: a 2-byte f16 scale + 32 int8 values.
        out_features, in_features = 8, 32
        block_bytes = 34
        raw = np.zeros((out_features, in_features // 32 * block_bytes), dtype=np.uint8)
        writer.add_tensor("v.patch_embd.weight", raw, raw_dtype=GGMLQuantizationType.Q8_0)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        mmproj_gguf = GGUFModel(str(path))
        report = _preflight_mmproj_quantization_report(mmproj_gguf, include_audio=False)

        assert len(report.tensor_records) == 1
        record = report.tensor_records[0]
        assert record.name == "mmproj:v.patch_embd.weight"
        assert record.qtype == "Q8_0"
        assert record.disposition is QuantizationDisposition.DEQUANTIZED_FLOAT
        assert record.target_storage == "float"
        assert report.source_fidelity is False
        assert report.storage_quantized is False

    def test_unknown_mapped_mmproj_qtype_fails_preflight(self) -> None:
        from mobius.integrations.gguf._mmproj import (
            _preflight_mmproj_quantization_report,
        )

        tensor = SimpleNamespace(
            name="v.patch_embd.weight",
            tensor_type=SimpleNamespace(name="UNKNOWN", value=99_999),
            shape=(8, 32),
            n_bytes=256,
        )
        source = SimpleNamespace(reader_tensors=lambda: iter([tensor]))

        with pytest.raises(ValueError, match=r"no safe disposition|pinned llama.cpp census"):
            _preflight_mmproj_quantization_report(source, include_audio=False)


class TestGemma4MultimodalQuantizationReportMerge:
    """The package-level report must combine text + mmproj without collisions.

    Covers the follow-up (issue 4): the Gemma4 multimodal builder previously
    preflighted only the text GGUF, so the persisted
    ``quantization_report.json`` silently excluded every mmproj vision/audio
    source tensor. These synthetic text+vision and text+vision/audio tests
    assert the merged package-level report's counts/bytes, tensor records,
    and JSON persistence.
    """

    def test_text_plus_vision_report_counts_bytes_records_and_persists(self, tmp_path: Path):
        from mobius._model_package import ModelPackage
        from mobius.integrations.gguf import build_gemma4_vlm_from_gguf
        from mobius.integrations.gguf._quantization_report import QuantizationDisposition
        from mobius.integrations.gguf._reader import GGUFModel

        mmproj_path = tmp_path / "mmproj.gguf"
        _write_clip_mmproj_gguf(mmproj_path, with_audio=True)
        text_path = tmp_path / "gemma4-q4.gguf"
        _write_quantized_gemma4_text_gguf(text_path)

        text_gguf = GGUFModel(str(text_path))
        mmproj_gguf = GGUFModel(str(mmproj_path))

        package = build_gemma4_vlm_from_gguf(text_path, mmproj_path, image_token_id=63)
        report = package.gguf_quantization_report

        # The census covers every tensor from BOTH source files (including
        # the mmproj's audio tensors, which are present but unmapped since
        # include_audio defaults to False).
        assert sum(stat.tensor_count for stat in report.source_qtype_census) == (
            text_gguf.num_tensors + mmproj_gguf.num_tensors
        )
        expected_bytes = sum(
            int(tensor.n_bytes) for tensor in text_gguf.reader_tensors()
        ) + sum(int(tensor.n_bytes) for tensor in mmproj_gguf.reader_tensors())
        assert sum(stat.source_bytes for stat in report.source_qtype_census) == expected_bytes

        # Mapped tensor records include both the text decoder's mapped
        # weights and the mmproj vision tower's, disambiguated by the
        # "mmproj:" prefix so the two components cannot collide.
        mmproj_records = [
            record for record in report.tensor_records if record.name.startswith("mmproj:")
        ]
        text_records = [
            record for record in report.tensor_records if not record.name.startswith("mmproj:")
        ]
        assert mmproj_records and text_records
        assert not any(record.name.startswith("mmproj:a.") for record in mmproj_records)
        assert all(
            record.disposition is QuantizationDisposition.SOURCE_FLOAT
            for record in mmproj_records
        )
        assert report.storage_quantized is True
        assert report.target_storage_format == "INT4 affine block-32 + float"

        # Persistence: the merged report round-trips through the package
        # save/load path exactly like the pre-existing single-source case.
        output_dir = tmp_path / "saved"
        package.save(str(output_dir), progress_bar=False)
        reloaded = ModelPackage.load(str(output_dir))
        assert reloaded.gguf_quantization_report == report

    def test_text_plus_vision_and_audio_reports_merge_counts_bytes_and_persist(
        self, tmp_path: Path
    ):
        """Exercises the include_audio=True mmproj preflight + merge.

        ``build_gemma4_vlm_from_gguf(..., include_audio=True)`` currently
        fails closed before reaching the quantization preflight step (the
        ``gemma4a`` audio projector is deferred/rejected -- see
        ``_preflight_mmproj_pair`` and the module docstring's audio caveat),
        so this drives the same mmproj preflight + merge helpers the builder
        itself uses at the level below that guard, covering the combined
        vision+audio report shape end to end. The "text" side is a minimal
        synthetic stand-in report (this test targets the mmproj preflight and
        the merge, which is the code changed for issue 4).
        """
        from mobius.integrations.gguf._mmproj import (
            _merge_component_quantization_reports,
            _preflight_mmproj_quantization_report,
        )
        from mobius.integrations.gguf._quantization_report import (
            GGUFQuantizationReport,
            QuantizationDisposition,
            QuantizationTensorRecord,
            disposition_for_import_route,
        )
        from mobius.integrations.gguf._reader import GGUFModel
        from mobius.integrations.gguf._spec import QuantImportRoute, RepackExactness

        mmproj_path = tmp_path / "mmproj.gguf"
        _write_clip_mmproj_gguf(mmproj_path, with_audio=True)
        mmproj_gguf = GGUFModel(str(mmproj_path))
        mmproj_report = _preflight_mmproj_quantization_report(mmproj_gguf, include_audio=True)

        # A minimal stand-in text-backbone component report (one lossless
        # native tensor, one float tensor) shaped like what
        # _builder._preflight_quantization_report produces for a real text
        # GGUF, so the merge is exercised against a genuinely quantized
        # component without needing to build a full Gemma4Model here.
        text_report = GGUFQuantizationReport.create(
            source_qtypes=[("Q4_0", 4608), ("F32", 256)],
            tensor_records=[
                QuantizationTensorRecord(
                    name="blk.0.attn_q.weight",
                    qtype="Q4_0",
                    source_bytes=4608,
                    disposition=disposition_for_import_route(
                        QuantImportRoute.NATIVE_BYTES, RepackExactness.EXACT
                    ),
                    target_storage="native GGUF block storage",
                    reason="synthetic stand-in for the text component",
                ),
                QuantizationTensorRecord(
                    name="token_embd.weight",
                    qtype="F32",
                    source_bytes=256,
                    disposition=QuantizationDisposition.SOURCE_FLOAT,
                    target_storage="float",
                    reason="synthetic stand-in for the text component",
                ),
            ],
            target_storage_format="native GGUF block storage",
            compute_mode="runtime-dependent native custom op or inline standard-ONNX fallback",
            compute_capability="synthetic stand-in capability",
        )

        merged = _merge_component_quantization_reports(text_report, mmproj_report)

        assert sum(stat.tensor_count for stat in merged.source_qtype_census) == sum(
            stat.tensor_count for stat in text_report.source_qtype_census
        ) + sum(stat.tensor_count for stat in mmproj_report.source_qtype_census)
        assert sum(stat.source_bytes for stat in merged.source_qtype_census) == sum(
            stat.source_bytes for stat in text_report.source_qtype_census
        ) + sum(stat.source_bytes for stat in mmproj_report.source_qtype_census)
        assert len(merged.tensor_records) == len(text_report.tensor_records) + len(
            mmproj_report.tensor_records
        )

        audio_records = [
            record
            for record in merged.tensor_records
            if record.name.startswith("mmproj:a.")
            or record.name == "mmproj:mm.a.input_projection.weight"
        ]
        vision_records = [
            record
            for record in merged.tensor_records
            if record.name.startswith("mmproj:v.")
            or record.name == "mmproj:mm.input_projection.weight"
        ]
        assert audio_records and vision_records
        assert {"blk.0.attn_q.weight", "token_embd.weight"} <= {
            record.name for record in merged.tensor_records
        }
        assert all(
            record.disposition is QuantizationDisposition.SOURCE_FLOAT
            for record in audio_records + vision_records
        )
        assert merged.storage_quantized is True
        assert merged.target_storage_format == "float + native GGUF block storage"

        # Persistence: the merged report round-trips through JSON exactly
        # like the package-level report does via ModelPackage.save/load.
        report_path = tmp_path / "quantization_report.json"
        merged.write_json(report_path)
        reloaded = GGUFQuantizationReport.read_json(report_path)
        assert reloaded == merged

    def test_merge_rejects_conflicting_duplicate_tensor_names(self):
        """Guard the source-name collision contract.

        If two component reports disagree about the same unqualified tensor
        name, the merge must fail loudly rather than silently pick one.
        """
        from mobius.integrations.gguf._mmproj import (
            _merge_component_quantization_reports,
        )
        from mobius.integrations.gguf._quantization_report import (
            GGUFQuantizationReport,
            QuantizationDisposition,
            QuantizationTensorRecord,
        )

        def make_report(disposition: QuantizationDisposition) -> GGUFQuantizationReport:
            return GGUFQuantizationReport.create(
                source_qtypes=[("F32", 4)],
                tensor_records=[
                    QuantizationTensorRecord(
                        name="shared.weight",
                        qtype="F32",
                        source_bytes=4,
                        disposition=disposition,
                        target_storage="float",
                        reason="conflict fixture",
                    )
                ],
                target_storage_format="float",
                compute_mode="float operators",
                compute_capability="test",
            )

        first = make_report(QuantizationDisposition.SOURCE_FLOAT)
        second = make_report(QuantizationDisposition.DEQUANTIZED_FLOAT)
        with pytest.raises(ValueError, match="Conflicting GGUF quantization dispositions"):
            _merge_component_quantization_reports(first, second)


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
            actual = build_vlm_from_gguf(
                text_path,
                mmproj_path,
                image_token_id=-200,
                keep_quantized=False,
            )

        assert actual is package
        builder.assert_called_once_with(
            str(text_path),
            str(mmproj_path),
            dtype=None,
            execution_provider="default",
            image_token_id=-200,
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
            image_token_id=None,
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
