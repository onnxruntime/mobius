# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for GGUF ``clip`` mmproj config extraction and vision-encoder build.

Builds a small synthetic ``clip`` mmproj GGUF with :class:`GGUFWriter` (mirroring
``_builder_test.py``), then exercises the mmproj config readers and the vision
encoder build+run path end-to-end on CPU.
"""

from __future__ import annotations

from pathlib import Path

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
_AUDIO_PROJ_OUT = 24


def _write_clip_mmproj_gguf(path: Path, *, with_audio: bool = True) -> None:
    """Write a small synthetic Gemma4 ``clip`` mmproj GGUF for tests."""
    from gguf import GGUFWriter

    head_dim = _VISION_HIDDEN // _VISION_HEADS
    writer = GGUFWriter(str(path), "clip")

    # --- vision metadata ---
    writer.add_bool("clip.has_vision_encoder", True)
    writer.add_string("clip.vision.projector_type", "gemma4v")
    writer.add_uint32("clip.vision.embedding_length", _VISION_HIDDEN)
    writer.add_uint32("clip.vision.feed_forward_length", _VISION_FFN)
    writer.add_uint32("clip.vision.block_count", _VISION_LAYERS)
    writer.add_uint32("clip.vision.attention.head_count", _VISION_HEADS)
    writer.add_uint32("clip.vision.image_size", _IMAGE_SIZE)
    writer.add_uint32("clip.vision.patch_size", _PATCH_SIZE)
    writer.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-6)

    def _f32(name: str, shape: tuple[int, ...]) -> None:
        writer.add_tensor(name, np.random.randn(*shape).astype(np.float32))

    # patch embed (conv layout) + position table + projector.
    _f32("v.patch_embd.weight", (_VISION_HIDDEN, 3, _PATCH_SIZE, _PATCH_SIZE))
    _f32("v.position_embd.weight", (2, _POS_EMB_SIZE, _VISION_HIDDEN))
    _f32("mm.input_projection.weight", (_TEXT_HIDDEN, _VISION_HIDDEN))
    # A companion activation-range stat tensor that must be skipped.
    _f32("mm.input_projection.weight.input_max", (1,))

    for layer in range(_VISION_LAYERS):
        prefix = f"v.blk.{layer}."
        for norm in ("ln1", "ln2", "attn_post_norm", "ffn_post_norm"):
            _f32(prefix + norm + ".weight", (_VISION_HIDDEN,))
        for proj in ("attn_q", "attn_k", "attn_v", "attn_out"):
            _f32(prefix + proj + ".weight", (_VISION_HIDDEN, _VISION_HIDDEN))
        for qk_norm in ("attn_q_norm", "attn_k_norm"):
            _f32(prefix + qk_norm + ".weight", (head_dim,))
        _f32(prefix + "ffn_gate.weight", (_VISION_FFN, _VISION_HIDDEN))
        _f32(prefix + "ffn_up.weight", (_VISION_FFN, _VISION_HIDDEN))
        _f32(prefix + "ffn_down.weight", (_VISION_HIDDEN, _VISION_FFN))
        # Stat tensor next to a quantizable linear — must be skipped.
        _f32(prefix + "attn_q.weight.output_min", (1,))

    # --- audio metadata (best-effort, for config-extraction tests) ---
    if with_audio:
        writer.add_bool("clip.has_audio_encoder", True)
        writer.add_string("clip.audio.projector_type", "gemma4a")
        writer.add_uint32("clip.audio.embedding_length", _AUDIO_HIDDEN)
        writer.add_uint32("clip.audio.feed_forward_length", _AUDIO_FFN)
        writer.add_uint32("clip.audio.block_count", _AUDIO_LAYERS)
        writer.add_uint32("clip.audio.attention.head_count", _AUDIO_HEADS)
        writer.add_uint32("clip.audio.num_mel_bins", _NUM_MEL_BINS)
        writer.add_float32("clip.audio.attention.layer_norm_epsilon", 1e-6)
        _f32("a.conv1d.0.weight", (_AUDIO_CONV0, 1, 3, 3))
        _f32("a.conv1d.1.weight", (_AUDIO_CONV1, _AUDIO_CONV0, 3, 3))
        _f32("mm.a.input_projection.weight", (_AUDIO_HIDDEN, _AUDIO_PROJ_OUT))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.fixture
def clip_mmproj_gguf(tmp_path: Path) -> Path:
    path = tmp_path / "mmproj.gguf"
    _write_clip_mmproj_gguf(path)
    return path


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
        assert config.pooling_kernel_size == 3
        assert config.position_embedding_size == _POS_EMB_SIZE
        assert config.use_clipped_linears is False
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


def test_special_token_id_uses_exact_token_match():
    from types import SimpleNamespace

    from mobius.integrations.gguf._mmproj import _special_token_id

    gguf = SimpleNamespace(
        metadata={
            "tokenizer.ggml.tokens": [
                "image",
                "<|image>",
                "<|image|>",
                "<|audio>",
                "<|audio|>",
            ]
        }
    )

    assert _special_token_id(gguf, "<|image|>") == 2
    assert _special_token_id(gguf, "<|audio|>") == 4


def test_mmproj_audio_expands_depthwise_conv_channel_dimension():
    from types import SimpleNamespace

    from mobius.integrations.gguf._mmproj import _mmproj_audio_to_hf

    values = np.ones((8, 5), dtype=np.float16)
    gguf = SimpleNamespace(
        tensor_names=["a.blk.0.conv_dw.weight"],
        get_tensor=lambda _name: values,
    )

    state_dict = _mmproj_audio_to_hf(gguf)

    assert state_dict["audio_tower.layers.0.lconv1d.depthwise_conv1d.weight"].shape == (
        8,
        1,
        5,
    )


def test_mmproj_audio_loads_activation_stats():
    from types import SimpleNamespace

    from mobius.integrations.gguf._mmproj import _mmproj_audio_to_hf

    values = np.array([-20.375], dtype=np.float32)
    gguf = SimpleNamespace(
        tensor_names=["a.blk.0.attn_q.input_min"],
        get_tensor=lambda _name: values,
    )

    state_dict = _mmproj_audio_to_hf(gguf)

    assert state_dict["audio_tower.layers.0.self_attn.q_proj.input_min"].shape == ()


class TestVisionEncoderBuildAndRun:
    """Build the Gemma4 vision encoder from the synthetic mmproj and run it."""

    def test_builds_applies_and_runs(self, clip_mmproj_gguf: Path, tmp_path: Path):
        import onnxruntime as ort
        import torch

        from mobius._configs import Gemma4Config
        from mobius._model_package import ModelPackage
        from mobius.integrations.gguf._mmproj import read_mmproj_vision_config
        from mobius.integrations.gguf._mmproj_mapping import map_mmproj_vision_to_hf
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
        state_dict: dict[str, torch.Tensor] = {}
        for name in gguf_model.tensor_names:
            if not (name.startswith("v.") or name == "mm.input_projection.weight"):
                continue
            hf_name = map_mmproj_vision_to_hf(name)
            if hf_name is None:
                continue
            values = np.array(gguf_model.get_tensor(name)).astype(np.float32)
            if name == "v.patch_embd.weight":
                values = values.reshape(values.shape[0], -1)
            state_dict[hf_name] = torch.from_numpy(values)

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

    def test_per_layer_input_projections_are_quantized_targets(self):
        from mobius.integrations.gguf._mmproj import _QUANTIZED_LINEAR_SUFFIXES

        assert ".per_layer_input_gate.weight" in _QUANTIZED_LINEAR_SUFFIXES
        assert ".per_layer_projection.weight" in _QUANTIZED_LINEAR_SUFFIXES

    def test_embedding_quantization_flag_controls_graph_layout(self, clip_mmproj_gguf: Path):
        import dataclasses

        from mobius._configs import Gemma4Config, QuantizationConfig
        from mobius.integrations.gguf._mmproj import read_mmproj_vision_config
        from mobius.integrations.gguf._reader import GGUFModel
        from mobius.models.gemma4 import Gemma4Model
        from mobius.tasks._gemma4 import Gemma4Task

        config = Gemma4Config(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=8,
            global_head_dim=16,
            vocab_size=64,
            layer_types=["full_attention"],
            intermediate_size=64,
            hidden_act="gelu_pytorch_tanh",
            vision=read_mmproj_vision_config(GGUFModel(str(clip_mmproj_gguf))),
            tie_word_embeddings=True,
        )
        config = dataclasses.replace(
            config,
            quantization=QuantizationConfig(
                bits=4,
                group_size=32,
                quant_method="gguf",
                sym=True,
                quantize_embeddings=False,
                quantize_lm_head=False,
            ),
        )

        package = Gemma4Task().build(Gemma4Model(config), config)

        assert "GatherBlockQuantized" not in _component_op_types(package["embedding"])
        assert [node.op_type for node in package["decoder"].graph].count("MatMulNBits") > 0

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
