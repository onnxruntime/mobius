# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests: build ONNX graphs for all supported architectures (no weights).

These tests verify that each model architecture can construct a valid ONNX
graph without downloading any weights. They are fast and require no network
access. Run with::

    pytest tests/build_graph_test.py -v

To run a single model::

    pytest tests/build_graph_test.py -k "qwen2"
"""

from __future__ import annotations

import re

import ml_dtypes
import numpy as np
import onnx_ir as ir
import pytest
from _test_configs import (
    ALL_CAUSAL_LM_CONFIGS,
    ALL_CONFIGS,
    AUTO_GENERATED_CONFIGS,
    DETECTION_CONFIGS,
    ENCODER_CONFIGS,
    LONGROPE_FACTORS,
    SEQ2SEQ_CONFIGS,
    SPEECH_CONFIGS,
    SSM_CONFIGS,
    TINY_HEAD_DIM,
    TINY_HEADS,
    TINY_HIDDEN,
    TINY_INTERMEDIATE,
    TINY_KV_HEADS,
    TINY_LAYERS,
    TINY_VOCAB,
    VISION_CONFIGS,
    VL_CONFIGS,
    _base_config,
    vl_overrides,
)

# --- ONNX Checker infrastructure (merged from onnx_checker_test.py) --------
from onnx_ir.passes.common import CheckerPass

from mobius._builder import (
    DTYPE_MAP,
    build_from_module,
)
from mobius._config_resolver import _default_task_for_model
from mobius._configs import (
    ArchitectureConfig,
    AudioConfig,
    CodePredictorConfig,
    MMSConfig,
    SpeakerEncoderConfig,
    TTSConfig,
    VisionConfig,
)
from mobius._optimizations import SymbolicShapeInferencePass
from mobius._pipeline_contract import component_presence, optional_input_contract
from mobius._registry import registry
from mobius.tasks import (
    CausalLMTask,
    Phi4MMMultiModalTask,
    Qwen3VLVisionLanguageTask,
    get_task,
)

_onnx_checker = CheckerPass()
_shape_inference = SymbolicShapeInferencePass()

# Models where the ONNX checker fails due to upstream onnx-ir issues
# (e.g. value_info missing type field for custom ops).
_CHECKER_SKIP_MODELS: set[str] = {
    "minimax",
    "qwen3_5_text",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "qwen3_next",
    # Models using LinearAttention / CausalConvWithState custom ops
    # prevent full shape/type propagation through com.microsoft domain.
    "bamba",
    "granitemoehybrid",
    "mamba2",
    "nemotron_h",
    "zamba2",
    # VL/Speech models with value_info/shape checker issues
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_vl_single",
    "qwen3_5_vl",
    "qwen3_5_moe_vl",
    "qwen3_tts_tokenizer_12hz",
}


def _fill_dummy_weights(model: ir.Model) -> None:
    """Fill initializers that have no const_value with zero tensors.

    Models built without weights leave initializers empty. The
    CheckerPass requires const_value to be set so it can serialize
    the model for the ONNX C checker.
    """
    for initializer in model.graph.initializers.values():
        if initializer.const_value is not None:
            continue
        shape = initializer.shape
        dims = [d if isinstance(d, int) else 1 for d in shape] if shape else [1]
        dtype = initializer.dtype or ir.DataType.FLOAT
        initializer.const_value = ir.Tensor(
            np.zeros(dims, dtype=dtype.numpy()),
        )


def _run_onnx_checker(pkg: dict[str, ir.Model], model_type: str) -> None:
    """Run ONNX CheckerPass on all models in a package.

    Skips models in ``_CHECKER_SKIP_MODELS`` that have known upstream issues.
    Runs shape inference first since the checker requires output shapes.
    """
    if model_type in _CHECKER_SKIP_MODELS:
        pytest.skip(
            f"ONNX checker skipped for {model_type}: "
            "upstream onnx-ir value_info missing type field for custom ops"
        )
    for model in pkg.values():
        _shape_inference(model)
        _fill_dummy_weights(model)
        _onnx_checker(model)


def _assert_outputs_have_shapes_and_dtypes(
    pkg: dict[str, ir.Model],
    model_type: str,
) -> None:
    """Assert every graph output has a non-None shape and dtype.

    Runs shape inference first (same as the real optimization pipeline)
    to populate output metadata, then verifies all outputs have both
    shape and type set.

    Skips models in ``_CHECKER_SKIP_MODELS`` whose custom ops prevent
    full shape propagation.
    """
    if model_type in _CHECKER_SKIP_MODELS:
        pytest.skip(
            f"Shape assertion skipped for {model_type}: "
            "custom ops prevent full shape propagation"
        )
    for sub_name, model in pkg.items():
        _shape_inference(model)
        for output in model.graph.outputs:
            assert output.shape is not None, (
                f"{model_type}/{sub_name}: output '{output.name}' "
                f"has no shape after shape inference"
            )
            assert output.type is not None, (
                f"{model_type}/{sub_name}: output '{output.name}' "
                f"has no dtype after shape inference"
            )


# Minimal configs for each architecture. These are hand-crafted small configs
# that exercise each model class without needing to download from HuggingFace.


# Semantic test IDs for model_types that intentionally appear more than once
# with different config overrides. Keyed by (model_type, occurrence_index).
_SEMANTIC_IDS: dict[tuple[str, int], str] = {
    ("deepseek_v2", 0): "deepseek_v2_mla",
    ("deepseek_v2", 1): "deepseek_v2_no_mla",
    ("deepseek_v2", 2): "deepseek_v2_mla_dense",
    ("qwen3_5_text", 0): "qwen3_5_text_default",
    ("qwen3_5_text", 1): "qwen3_5_text_linear_attn",
    ("qwen3_next", 0): "qwen3_next_hybrid",
    ("qwen3_next", 1): "qwen3_next_all_full_attn",
    ("qwen3_next", 2): "qwen3_next_all_linear_attn",
    ("falcon_h1", 0): "falcon_h1_alibi",
    ("falcon_h1", 1): "falcon_h1_parallel_attn",
    ("jamba", 0): "jamba_hybrid_moe",
    ("jamba", 1): "jamba_all_attention",
    ("bamba", 0): "bamba_hybrid",
    ("bamba", 1): "bamba_all_attention",
    ("gemma3n_text", 0): "gemma3n_text_sliding",
    ("gemma3n_text", 1): "gemma3n_text_full_attn",
    ("granite", 0): "granite_default",
    ("granite", 1): "granite_scaling",
    ("phi3small", 0): "phi3small_default",
    ("phi3small", 1): "phi3small_rotary_025",
}


def _make_params(
    configs: list[tuple[str, dict, bool]],
) -> list:
    """Create pytest.param entries with stable unique IDs.

    Duplicate model_types get semantic IDs from ``_SEMANTIC_IDS`` when
    available, falling back to ``<model_type>_<index>`` otherwise.
    """
    from collections import Counter

    stripped = [(mt, ov) for mt, ov, _ in configs]
    counts = Counter(mt for mt, _ in stripped)
    seen: dict[str, int] = {}
    params = []
    for model_type, overrides in stripped:
        if counts[model_type] > 1:
            idx = seen.get(model_type, 0)
            seen[model_type] = idx + 1
            test_id = _SEMANTIC_IDS.get((model_type, idx), f"{model_type}_{idx}")
        else:
            test_id = model_type
        params.append(pytest.param(model_type, overrides, id=test_id))
    return params


# Configs imported from _test_configs — strip the is_representative flag
# for use with pytest.parametrize.
_MODEL_CONFIGS: list[tuple[str, dict]] = [(mt, ov) for mt, ov, _ in ALL_CAUSAL_LM_CONFIGS]

_MODEL_PARAMS = _make_params(ALL_CAUSAL_LM_CONFIGS)


@pytest.mark.parametrize("model_type,config_overrides", _MODEL_PARAMS)
class TestBuildGraph:
    """Verify that each model type builds a valid ONNX graph."""

    def test_graph_builds_without_weights(self, model_type: str, config_overrides: dict):
        """Build a model graph from a tiny config and verify basic structure."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)
        model = pkg["model"]

        # Basic structure checks
        assert model.graph is not None
        assert len(model.graph.inputs) > 0, "Model should have inputs"
        assert len(model.graph.outputs) > 0, "Model should have outputs"

        # Check expected inputs exist
        input_names = {inp.name for inp in model.graph.inputs}
        assert "input_ids" in input_names
        assert "attention_mask" in input_names
        assert "position_ids" in input_names

        # Check outputs include logits and KV cache
        output_names = {out.name for out in model.graph.outputs}
        assert "logits" in output_names

        # Check KV cache / hybrid cache outputs.  Models whose trailing layers
        # borrow K,V from an earlier layer (Gemma 3n's num_kv_shared_layers)
        # expose fewer cache entries than they have layers; the non-shared
        # layers are the leading ones, so truncating the range is enough.
        count_fn = getattr(module, "kv_cache_layer_count", None)
        num_layers = count_fn() if callable(count_fn) else config.num_hidden_layers
        layer_types = config.layer_types or []
        for i in range(num_layers):
            ltype = layer_types[i] if i < len(layer_types) else "full_attention"
            if ltype in ("mlp", "moe"):
                continue  # MLP and MoE layers are stateless — no cache outputs
            if ltype == "lightning_attention":
                # Lightning Attention: single recurrent state only (no conv_state)
                assert f"present.{i}.recurrent_state" in output_names, (
                    f"Missing present.{i}.recurrent_state"
                )
            elif ltype in ("linear_attention",):
                assert f"present.{i}.conv_state" in output_names, (
                    f"Missing present.{i}.conv_state"
                )
                assert f"present.{i}.recurrent_state" in output_names, (
                    f"Missing present.{i}.recurrent_state"
                )
            elif ltype in ("mamba", "mamba2"):
                assert f"present.{i}.conv_state" in output_names, (
                    f"Missing present.{i}.conv_state"
                )
                assert f"present.{i}.ssm_state" in output_names, (
                    f"Missing present.{i}.ssm_state"
                )
            elif ltype == "conv":
                assert f"present.{i}.conv_state" in output_names, (
                    f"Missing present.{i}.conv_state"
                )
            else:
                assert f"present.{i}.key" in output_names, f"Missing present.{i}.key"
                assert f"present.{i}.value" in output_names, f"Missing present.{i}.value"

    def test_graph_has_initializers(self, model_type: str, config_overrides: dict):
        """Verify the graph has initializers (parameters) even without weight values."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)
        model = pkg["model"]

        init_names = list(model.graph.initializers)
        assert len(init_names) > 0, "Model should have initializers"

        # Check for expected parameter patterns (allow model-specific naming)
        has_embed = any(
            "embed_tokens" in n or "word_embeddings" in n or "wte" in n or "embed_in" in n
            for n in init_names
        )
        has_attn = any(
            "self_attn" in n or "self_attention" in n or "attention" in n or ".attn." in n
            for n in init_names
        )
        has_mlp = any("mlp" in n or "expert" in n or "feed_forward" in n for n in init_names)
        assert has_embed, "Should have embedding parameters"
        assert has_attn, "Should have attention parameters"
        assert has_mlp, "Should have MLP parameters"

    def test_onnx_checker_passes(self, model_type: str, config_overrides: dict):
        """Run the ONNX CheckerPass to catch attribute/shape/type errors."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = get_task(_default_task_for_model(model_type))
        pkg = task.build(module, config)
        _run_onnx_checker(pkg, model_type)

    def test_outputs_have_shapes_and_dtypes(self, model_type: str, config_overrides: dict):
        """Verify shape inference populates all output shapes and dtypes."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = get_task(_default_task_for_model(model_type))
        pkg = task.build(module, config)
        _assert_outputs_have_shapes_and_dtypes(pkg, model_type)


class TestTextDecoderBatchGreaterThanOne:
    """Text-only causal LM decoders must run with ``batch_size > 1``.

    Regression guard for the attention-mask head-dim bug: ``create_padding_mask``
    / ``create_sliding_window_mask`` previously returned a 3-D ``(B, q, total)``
    mask whose batch axis was right-aligned onto ``q_num_heads`` by the ONNX
    Attention op — it worked for ``batch == 1`` but ORT rejected ``batch > 1``.
    Covers a plain decoder, GQA, and sliding-window architectures, with ragged
    padding across rows.
    """

    @pytest.mark.parametrize("model_type", ["qwen2", "llama", "mistral", "gemma2"])
    def test_batch2_prefill_runs(self, model_type: str):
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.rewrite_rules._testing_utils import fill_random_weights

        overrides = dict(_MODEL_CONFIGS)[model_type]
        config = _base_config(**overrides)
        module = registry.get(model_type)(config)
        task = get_task(_default_task_for_model(model_type))
        pkg = task.build(module, config)
        model = pkg["model"]
        fill_random_weights(model)
        sess = OnnxModelSession(model)

        batch, seq = 2, 6
        rng = np.random.default_rng(0)
        input_ids = rng.integers(1, config.vocab_size, size=(batch, seq)).astype(np.int64)
        attention_mask = np.ones((batch, seq), dtype=np.int64)
        attention_mask[1, :2] = 0  # row 1 has two leading padding tokens
        position_ids = np.tile(np.arange(seq, dtype=np.int64), (batch, 1))
        feeds: dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        for i in range(config.num_hidden_layers):
            feeds[f"past_key_values.{i}.key"] = np.zeros(
                (batch, config.num_key_value_heads, 0, config.head_dim), dtype=np.float32
            )
            feeds[f"past_key_values.{i}.value"] = np.zeros(
                (batch, config.num_key_value_heads, 0, config.head_dim), dtype=np.float32
            )

        out = sess.run(feeds)
        sess.close()
        logits = out["logits"]
        assert logits.shape[0] == batch
        assert logits.shape[1] == seq
        # Independent rows (different inputs) must produce different logits.
        assert not np.allclose(logits[0], logits[1])


# === Encoder-only model configs (imported from _test_configs) ===
_ENCODER_MODEL_CONFIGS: list[tuple[str, dict]] = [(mt, ov) for mt, ov, _ in ENCODER_CONFIGS]

_ENCODER_MODEL_PARAMS = _make_params(ENCODER_CONFIGS)


@pytest.mark.parametrize("model_type,config_overrides", _ENCODER_MODEL_PARAMS)
class TestBuildEncoderGraph:
    """Verify that encoder-only model types build valid ONNX graphs."""

    def test_graph_builds_without_weights(self, model_type: str, config_overrides: dict):
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "input_ids" in input_names
        assert "attention_mask" in input_names
        assert "token_type_ids" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "last_hidden_state" in output_names
        # No KV cache outputs for encoder-only models
        assert not any(n.startswith("present.") for n in output_names)

    def test_graph_has_initializers(self, model_type: str, config_overrides: dict):
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)
        model = pkg["model"]

        init_names = list(model.graph.initializers)
        assert len(init_names) > 0
        has_embed = any("word_embeddings" in n or "embed" in n for n in init_names)
        has_attn = any(
            "self_attn" in n or "self_attention" in n or "attention" in n or ".attn." in n
            for n in init_names
        )
        has_mlp = any(
            "mlp" in n or "ffn" in n or "feed_forward" in n or "intermediate" in n
            for n in init_names
        )
        assert has_embed, "Should have word embedding parameters"
        assert has_attn, "Should have attention parameters"
        assert has_mlp, "Should have MLP parameters"

    def test_onnx_checker_passes(self, model_type: str, config_overrides: dict):
        """Run the ONNX CheckerPass to catch attribute/shape/type errors."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = get_task(_default_task_for_model(model_type))
        pkg = task.build(module, config)
        _run_onnx_checker(pkg, model_type)

    def test_outputs_have_shapes_and_dtypes(self, model_type: str, config_overrides: dict):
        """Verify shape inference populates all output shapes and dtypes."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = get_task(_default_task_for_model(model_type))
        pkg = task.build(module, config)
        _assert_outputs_have_shapes_and_dtypes(pkg, model_type)


# === Encoder-decoder model configs (imported from _test_configs) ===
_SEQ2SEQ_MODEL_CONFIGS: list[tuple[str, dict]] = [(mt, ov) for mt, ov, _ in SEQ2SEQ_CONFIGS]

_SEQ2SEQ_MODEL_PARAMS = _make_params(SEQ2SEQ_CONFIGS)


@pytest.mark.parametrize("model_type,config_overrides", _SEQ2SEQ_MODEL_PARAMS)
class TestBuildSeq2SeqGraph:
    """Verify that encoder-decoder model types build valid ONNX graphs."""

    def test_encoder_graph_builds(self, model_type: str, config_overrides: dict):
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)
        model = pkg["encoder"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "input_ids" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "last_hidden_state" in output_names

    def test_package_has_encoder_and_decoder(self, model_type: str, config_overrides: dict):
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert "encoder" in pkg
        assert "decoder" in pkg

        dec_outputs = {out.name for out in pkg["decoder"].graph.outputs}
        assert "logits" in dec_outputs

    def test_onnx_checker_passes(self, model_type: str, config_overrides: dict):
        """Run the ONNX CheckerPass to catch attribute/shape/type errors."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = get_task(_default_task_for_model(model_type))
        pkg = task.build(module, config)
        _run_onnx_checker(pkg, model_type)

    def test_outputs_have_shapes_and_dtypes(self, model_type: str, config_overrides: dict):
        """Verify shape inference populates all output shapes and dtypes."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = get_task(_default_task_for_model(model_type))
        pkg = task.build(module, config)
        _assert_outputs_have_shapes_and_dtypes(pkg, model_type)


# === Vision model configs (imported from _test_configs) ===
_VISION_MODEL_CONFIGS: list[tuple[str, dict]] = [(mt, ov) for mt, ov, _ in VISION_CONFIGS]

_VISION_MODEL_PARAMS = _make_params(VISION_CONFIGS)


@pytest.mark.parametrize("model_type,config_overrides", _VISION_MODEL_PARAMS)
class TestBuildVisionGraph:
    """Verify that vision model types build valid ONNX graphs."""

    def test_graph_builds_without_weights(self, model_type: str, config_overrides: dict):
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "pixel_values" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "last_hidden_state" in output_names

    def test_onnx_checker_passes(self, model_type: str, config_overrides: dict):
        """Run the ONNX CheckerPass to catch attribute/shape/type errors."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = get_task(_default_task_for_model(model_type))
        pkg = task.build(module, config)
        _run_onnx_checker(pkg, model_type)

    def test_outputs_have_shapes_and_dtypes(self, model_type: str, config_overrides: dict):
        """Verify shape inference populates all output shapes and dtypes."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = get_task(_default_task_for_model(model_type))
        pkg = task.build(module, config)
        _assert_outputs_have_shapes_and_dtypes(pkg, model_type)


# === Object detection model configs (imported from _test_configs) ===
_DETECTION_MODEL_CONFIGS: list[tuple[str, dict]] = [
    (mt, ov) for mt, ov, _ in DETECTION_CONFIGS
]

_DETECTION_MODEL_PARAMS = _make_params(DETECTION_CONFIGS)


@pytest.mark.parametrize("model_type,config_overrides", _DETECTION_MODEL_PARAMS)
class TestBuildDetectionGraph:
    """Verify that object detection model types build valid ONNX graphs."""

    def test_graph_builds_without_weights(self, model_type: str, config_overrides: dict):
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "pixel_values" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "logits" in output_names
        assert "pred_boxes" in output_names

    def test_onnx_checker_passes(self, model_type: str, config_overrides: dict):
        """Run the ONNX CheckerPass to catch attribute/shape/type errors."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = get_task(_default_task_for_model(model_type))
        pkg = task.build(module, config)
        _run_onnx_checker(pkg, model_type)

    def test_outputs_have_shapes_and_dtypes(self, model_type: str, config_overrides: dict):
        """Verify shape inference populates all output shapes and dtypes."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = get_task(_default_task_for_model(model_type))
        pkg = task.build(module, config)
        _assert_outputs_have_shapes_and_dtypes(pkg, model_type)


# === SSM (Mamba/Mamba2) configs ===
_SSM_MODEL_PARAMS = _make_params(SSM_CONFIGS)


@pytest.mark.parametrize("model_type,config_overrides", _SSM_MODEL_PARAMS)
class TestBuildSSMGraph:
    """Verify that SSM (Mamba/Mamba2) model types build valid ONNX graphs."""

    def test_graph_builds_without_weights(self, model_type: str, config_overrides: dict):
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "input_ids" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "logits" in output_names

        # SSM models carry conv_state + ssm_state per layer
        num_layers = config.num_hidden_layers
        for i in range(num_layers):
            assert f"present.{i}.conv_state" in output_names
            assert f"present.{i}.ssm_state" in output_names

    def test_graph_has_initializers(self, model_type: str, config_overrides: dict):
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)
        model = pkg["model"]

        init_names = list(model.graph.initializers)
        assert len(init_names) > 0, "Model should have initializers"

        has_embed = any("embeddings" in n for n in init_names)
        has_mixer = any("mixer" in n for n in init_names)
        has_norm = any("norm" in n for n in init_names)
        assert has_embed, "Should have embedding parameters"
        assert has_mixer, "Should have mixer (SSM) parameters"
        assert has_norm, "Should have norm parameters"

    def test_onnx_checker_passes(self, model_type: str, config_overrides: dict):
        """Run the ONNX CheckerPass to catch attribute/shape/type errors."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = get_task(_default_task_for_model(model_type))
        pkg = task.build(module, config)
        _run_onnx_checker(pkg, model_type)

    def test_outputs_have_shapes_and_dtypes(self, model_type: str, config_overrides: dict):
        """Verify shape inference populates all output shapes and dtypes."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = get_task(_default_task_for_model(model_type))
        pkg = task.build(module, config)
        _assert_outputs_have_shapes_and_dtypes(pkg, model_type)


class TestBuildGraphLoRA:
    """Verify LoRA-specific structure in Phi4MM graph."""

    def _phi4mm_config(self):
        return _base_config(
            partial_rotary_factor=0.5,
            rope_type="longrope",
            rope_scaling={
                "short_factor": LONGROPE_FACTORS,
                "long_factor": LONGROPE_FACTORS,
            },
            original_max_position_embeddings=128,
            vision=VisionConfig(
                lora={"r": 4, "lora_alpha": 8},
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
                norm_eps=1e-6,
            ),
            audio=AudioConfig(
                lora={"r": 8, "lora_alpha": 16},
                attention_dim=32,
                attention_heads=2,
                num_blocks=1,
                linear_units=64,
                kernel_size=3,
                input_size=16,
                conv_channels=32,
                t5_bias_max_distance=10,
            ),
            image_token_id=200010,
        )

    def test_lora_initializers_present(self):
        config = self._phi4mm_config()
        model_cls = registry.get("phi4mm")
        module = model_cls(config)
        task = Phi4MMMultiModalTask()
        pkg = task.build(module, config)
        # LoRA adapters live in the decoder model (pkg["decoder"])
        decoder = pkg["decoder"]

        init_names = list(decoder.graph.initializers)
        lora_names = [n for n in init_names if "lora" in n]
        assert len(lora_names) > 0, "Phi4MM should have LoRA initializers"

        # Each layer should have LoRA for q/k/v/o_proj and gate/up/down_proj
        # Each proj has lora_A and lora_B for both vision and speech adapters
        vision_a = [n for n in lora_names if "lora_A.vision" in n]
        vision_b = [n for n in lora_names if "lora_B.vision" in n]
        speech_a = [n for n in lora_names if "lora_A.speech" in n]
        speech_b = [n for n in lora_names if "lora_B.speech" in n]
        assert len(vision_a) > 0, "Should have vision lora_A"
        assert len(vision_b) > 0, "Should have vision lora_B"
        assert len(speech_a) > 0, "Should have speech lora_A"
        assert len(speech_b) > 0, "Should have speech lora_B"


class TestBuildGraphQuantized:
    """Verify quantized model graphs use MatMulNBits."""

    TINY_LAYERS = 2
    NUM_PROJECTIONS_PER_LAYER = 7  # q, k, v, o, gate, up, down

    def _quantized_config(self, sym=True):
        from mobius._configs import QuantizationConfig

        qc = QuantizationConfig(bits=4, group_size=32, quant_method="gptq", sym=sym)
        return _base_config(
            num_hidden_layers=self.TINY_LAYERS,
            quantization=qc,
        )

    def test_matmulnbits_count(self):
        """Each layer has 7 projections → 2 layers = 14 MatMulNBits ops."""
        config = self._quantized_config()
        model_cls = registry.get("llama")
        module = model_cls(config)
        task = CausalLMTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        matmulnbits = [n for n in model.graph if n.op_type == "MatMulNBits"]
        expected = self.TINY_LAYERS * self.NUM_PROJECTIONS_PER_LAYER
        assert len(matmulnbits) == expected, (
            f"Expected {expected} MatMulNBits, got {len(matmulnbits)}"
        )

    def test_scales_initializers_present(self):
        """Quantized projections should have scales initializers."""
        config = self._quantized_config()
        model_cls = registry.get("llama")
        module = model_cls(config)
        task = CausalLMTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        init_names = list(model.graph.initializers)
        scales_names = [n for n in init_names if ".scales" in n]
        expected = self.TINY_LAYERS * self.NUM_PROJECTIONS_PER_LAYER
        assert len(scales_names) == expected, (
            f"Expected {expected} scales initializers, got {len(scales_names)}"
        )

    def test_asymmetric_has_zero_points(self):
        """Asymmetric quantization should have zero_points initializers."""
        config = self._quantized_config(sym=False)
        model_cls = registry.get("llama")
        module = model_cls(config)
        task = CausalLMTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        init_names = list(model.graph.initializers)
        zp_names = [n for n in init_names if ".zero_points" in n]
        expected = self.TINY_LAYERS * self.NUM_PROJECTIONS_PER_LAYER
        assert len(zp_names) == expected, (
            f"Expected {expected} zero_points, got {len(zp_names)}"
        )

    def test_lm_head_stays_fp(self):
        """lm_head should remain a standard Linear (MatMul), not quantized."""
        config = self._quantized_config()
        model_cls = registry.get("llama")
        module = model_cls(config)
        assert type(module.lm_head).__name__ == "Linear"

    def test_no_quantization_no_matmulnbits(self):
        """Without quantization config, no MatMulNBits ops should exist."""
        config = _base_config(num_hidden_layers=1)
        model_cls = registry.get("llama")
        module = model_cls(config)
        task = CausalLMTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        matmulnbits = [n for n in model.graph if n.op_type == "MatMulNBits"]
        assert len(matmulnbits) == 0, "Non-quantized model should have no MatMulNBits"

    def test_awq_produces_matmulnbits(self):
        """AWQ quantization should also produce MatMulNBits ops."""
        from mobius._configs import QuantizationConfig

        qc = QuantizationConfig(bits=4, group_size=32, quant_method="awq", sym=False)
        config = _base_config(num_hidden_layers=1, quantization=qc)
        model_cls = registry.get("llama")
        module = model_cls(config)
        task = CausalLMTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        matmulnbits = [n for n in model.graph if n.op_type == "MatMulNBits"]
        expected = 1 * self.NUM_PROJECTIONS_PER_LAYER
        assert len(matmulnbits) == expected


class TestBuildGraphVisionLanguage:
    """Verify multimodal models build correctly."""

    def test_phi4mm_multimodal_graph(self):
        """Build Phi4MM with Phi4MMMultiModalTask and verify 4-model split."""
        config = _base_config(
            partial_rotary_factor=0.5,
            rope_type="longrope",
            rope_scaling={
                "short_factor": LONGROPE_FACTORS,
                "long_factor": LONGROPE_FACTORS,
            },
            original_max_position_embeddings=128,
            vision=VisionConfig(
                lora={"r": 4, "lora_alpha": 8},
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
                norm_eps=1e-6,
            ),
            audio=AudioConfig(
                lora={"r": 8, "lora_alpha": 16},
                attention_dim=32,
                attention_heads=2,
                num_blocks=1,
                linear_units=64,
                kernel_size=3,
                input_size=16,
                conv_channels=32,
                t5_bias_max_distance=10,
            ),
            image_token_id=200010,
        )
        model_cls = registry.get("phi4mm")
        module = model_cls(config)
        task = Phi4MMMultiModalTask()
        pkg = task.build(module, config)

        # Verify 4-model package structure
        assert "vision_encoder" in pkg, "Should have vision model"
        assert "audio_encoder" in pkg, "Should have audio model"
        assert "embedding" in pkg, "Should have embedding model"
        assert "decoder" in pkg, "Should have decoder model"

        # Vision model: pixel_values + image_sizes → image_features
        vision = pkg["vision_encoder"]
        v_inputs = {inp.name for inp in vision.graph.inputs}
        v_outputs = {out.name for out in vision.graph.outputs}
        assert "pixel_values" in v_inputs
        assert "image_sizes" in v_inputs
        assert "image_features" in v_outputs
        v_inits = list(vision.graph.initializers)
        assert any("img_processor" in n for n in v_inits), (
            "Vision model should have SigLIP initializers"
        )

        # Speech model: audio_embeds + metadata → audio_features (single output)
        speech = pkg["audio_encoder"]
        s_inputs = {inp.name for inp in speech.graph.inputs}
        s_outputs = {out.name for out in speech.graph.outputs}
        assert "audio_embeds" in s_inputs
        assert "audio_sizes" in s_inputs
        assert "audio_projection_mode" in s_inputs
        assert "audio_features" in s_outputs

        # Embedding model: input_ids + features → inputs_embeds
        emb = pkg["embedding"]
        e_inputs = {inp.name for inp in emb.graph.inputs}
        e_outputs = {out.name for out in emb.graph.outputs}
        assert "input_ids" in e_inputs
        assert "image_features" in e_inputs
        assert "audio_features" in e_inputs
        assert "inputs_embeds" in e_outputs

        # Decoder model (pkg["decoder"]): inputs_embeds → logits + KV cache
        decoder = pkg["decoder"]
        d_inputs = {inp.name for inp in decoder.graph.inputs}
        d_outputs = {out.name for out in decoder.graph.outputs}
        assert "inputs_embeds" in d_inputs
        assert "attention_mask" in d_inputs
        assert "position_ids" in d_inputs
        assert "logits" in d_outputs

    def test_llava_vision_language_graph(self):
        """Build LLaVA with 3-model split and verify all components."""
        config = _base_config(
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
                norm_eps=1e-6,
            ),
            image_token_id=32000,
        )
        model_cls = registry.get("llava")
        module = model_cls(config)
        task_name = _default_task_for_model("llava")
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}

        decoder = pkg["decoder"]
        assert "inputs_embeds" in {i.name for i in decoder.graph.inputs}
        assert "logits" in {o.name for o in decoder.graph.outputs}

        vision = pkg["vision_encoder"]
        assert "pixel_values" in {i.name for i in vision.graph.inputs}
        assert "image_features" in {o.name for o in vision.graph.outputs}

        embed = pkg["embedding"]
        assert "input_ids" in {i.name for i in embed.graph.inputs}
        assert "inputs_embeds" in {o.name for o in embed.graph.outputs}

    def test_internvl2_vision_language_graph(self):
        """Build InternVL2 with 3-model split and verify all components."""
        config = _base_config(
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
                norm_eps=1e-6,
            ),
            image_token_id=32000,
        )
        model_cls = registry.get("internvl_chat")
        module = model_cls(config)
        task_name = _default_task_for_model("internvl_chat")
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}

        decoder = pkg["decoder"]
        assert "inputs_embeds" in {i.name for i in decoder.graph.inputs}
        assert "logits" in {o.name for o in decoder.graph.outputs}

        vision = pkg["vision_encoder"]
        assert "pixel_values" in {i.name for i in vision.graph.inputs}
        assert "image_features" in {o.name for o in vision.graph.outputs}

        embed = pkg["embedding"]
        assert "input_ids" in {i.name for i in embed.graph.inputs}
        assert "inputs_embeds" in {o.name for o in embed.graph.outputs}

        # Verify aliases also resolve to InternVL2Model
        from mobius.models.internvl import InternVL2Model

        for alias in ("internvl2", "internvl"):
            alias_cls = registry.get(alias)
            assert alias_cls is InternVL2Model, f"{alias} should map to InternVL2Model"

    def test_qwen2_5_vl_graph(self):
        """Build Qwen2.5-VL with its auto-detected 3-model task."""
        config = _base_config(
            attn_qkv_bias=True,
            mrope_section=[8, 12, 12],
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                patch_size=14,
                in_channels=3,
                out_hidden_size=64,
            ),
            temporal_patch_size=2,
            spatial_merge_size=2,
            fullatt_block_indexes=[1],
            image_token_id=151655,
        )
        model_cls = registry.get("qwen2_5_vl")
        module = model_cls(config)
        task_name = _default_task_for_model("qwen2_5_vl")
        task = get_task(task_name)
        pkg = task.build(module, config)

        # 3-model split: decoder, vision, embedding
        assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}

        # Decoder: inputs_embeds → logits + KV cache
        decoder = pkg["decoder"]
        assert "inputs_embeds" in {i.name for i in decoder.graph.inputs}
        assert "logits" in {o.name for o in decoder.graph.outputs}

        # Vision: pixel_values → image_features
        vision = pkg["vision_encoder"]
        assert "pixel_values" in {i.name for i in vision.graph.inputs}
        assert "image_features" in {o.name for o in vision.graph.outputs}

        # Embedding: input_ids + image_features → inputs_embeds
        embed = pkg["embedding"]
        assert "input_ids" in {i.name for i in embed.graph.inputs}
        assert "inputs_embeds" in {o.name for o in embed.graph.outputs}

    def test_qwen2_5_vl_text_graph(self):
        """Build Qwen2.5-VL text-only model."""
        config = _base_config(attn_qkv_bias=True, mrope_section=[8, 12, 12])
        model_cls = registry.get("qwen2_5_vl_text")
        module = model_cls(config)
        task_name = _default_task_for_model("qwen2_5_vl_text")
        task = get_task(task_name)
        pkg = task.build(module, config)
        model = pkg["model"]
        assert model.graph is not None
        assert "logits" in {out.name for out in model.graph.outputs}

    def test_qwen3_vl_graph(self):
        """Build Qwen3-VL with its auto-detected 3-model task."""
        config = _base_config(
            attn_qk_norm=True,
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                patch_size=16,
                in_channels=3,
                out_hidden_size=64,
                num_position_embeddings=16,
            ),
            temporal_patch_size=2,
            spatial_merge_size=2,
            deepstack_visual_indexes=[0],
            image_token_id=151655,
            mrope_section=[8, 12, 12],
        )
        model_cls = registry.get("qwen3_vl")
        module = model_cls(config)
        task_name = _default_task_for_model("qwen3_vl")
        task = get_task(task_name)
        pkg = task.build(module, config)

        # 3-model split produces decoder, vision, embedding
        assert "decoder" in pkg
        assert "vision_encoder" in pkg
        assert "embedding" in pkg

        # Decoder should have logits output and inputs_embeds input
        decoder = pkg["decoder"]
        assert "logits" in {out.name for out in decoder.graph.outputs}
        assert "inputs_embeds" in {inp.name for inp in decoder.graph.inputs}

    def test_qwen35_vl_graph(self):
        """Build Qwen3.5-VL with its auto-detected 3-model task."""
        config = _base_config(
            attn_qk_norm=True,
            partial_rotary_factor=0.5,
            layer_types=["linear_attention", "full_attention"],
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                patch_size=16,
                in_channels=3,
                out_hidden_size=64,
                num_position_embeddings=16,
            ),
            temporal_patch_size=2,
            spatial_merge_size=2,
            deepstack_visual_indexes=[0],
            image_token_id=248056,
            mrope_section=[8, 12, 12],
        )
        model_cls = registry.get("qwen3_5_vl")
        module = model_cls(config)
        task_name = _default_task_for_model("qwen3_5_vl")
        task = get_task(task_name)
        pkg = task.build(module, config)

        # 3-model split produces decoder, vision, embedding
        assert "decoder" in pkg
        assert "vision_encoder" in pkg
        assert "embedding" in pkg

        # Decoder should have logits output and inputs_embeds input
        decoder = pkg["decoder"]
        assert "logits" in {out.name for out in decoder.graph.outputs}
        assert "inputs_embeds" in {inp.name for inp in decoder.graph.inputs}

        # Verify hybrid cache: linear_attention layer gets conv_state/recurrent_state,
        # full_attention layer gets key/value
        output_names = {out.name for out in decoder.graph.outputs}
        assert "present.0.conv_state" in output_names
        assert "present.0.recurrent_state" in output_names
        assert "present.1.key" in output_names
        assert "present.1.value" in output_names

    def test_qwen3_vl_single_model_graph(self):
        """Build Qwen3-VL with single-model Qwen3VLVisionLanguageTask."""
        config = _base_config(
            attn_qk_norm=True,
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                patch_size=16,
                in_channels=3,
                out_hidden_size=64,
                num_position_embeddings=16,
            ),
            temporal_patch_size=2,
            spatial_merge_size=2,
            deepstack_visual_indexes=[0],
            image_token_id=151655,
            mrope_section=[8, 12, 12],
        )
        model_cls = registry.get("qwen3_vl_single")
        module = model_cls(config)
        task = Qwen3VLVisionLanguageTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        assert model.graph is not None
        assert "logits" in {out.name for out in model.graph.outputs}

    def test_gemma3_multimodal_graph(self):
        """Build Gemma3 multimodal model with 3-model split."""
        config = _base_config(
            attn_qk_norm=True,
            rope_local_base_freq=10_000.0,
            layer_types=["full_attention", "sliding_attention"],
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
                norm_eps=1e-6,
            ),
            mm_tokens_per_image=4,
            image_token_id=255999,
        )
        model_cls = registry.get("gemma3")
        module = model_cls(config)
        task_name = _default_task_for_model("gemma3")
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}
        assert "pixel_values" in {i.name for i in pkg["vision_encoder"].graph.inputs}
        assert "logits" in {o.name for o in pkg["decoder"].graph.outputs}

    def test_gemma4_multimodal_graph(self):
        """Build Gemma4 vision-language model via registry (3-model split: decoder+vision+embedding).

        The ``gemma4`` model type maps to Gemma4Model.  Without an audio config,
        the package has three models: decoder, vision, embedding.
        """
        from mobius._configs import Gemma4Config

        config = Gemma4Config(
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=16,
            vocab_size=256,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            attn_qk_norm=True,
            # Dual layer types: 1 local + 1 global
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=8,
            # Global attention config (same head_dim in test for simplicity)
            global_head_dim=16,
            global_rope_theta=10_000.0,
            global_partial_rotary_factor=0.25,
            final_logit_softcapping=0.0,
            hidden_size_per_layer_input=0,
            image_token_id=255999,
            pad_token_id=0,
            tie_word_embeddings=True,
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                patch_size=16,
                norm_eps=1e-6,
            ),
        )
        model_cls = registry.get("gemma4")
        module = model_cls(config)
        task_name = _default_task_for_model("gemma4")
        task = get_task(task_name)
        pkg = build_from_module(module, config, task=task)

        assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}, (
            f"Vision-only Gemma4 should produce 3 models, got: {set(pkg.keys())}"
        )
        # Decoder: inputs_embeds -> logits + per-layer KV cache
        decoder = pkg["decoder"]
        assert "inputs_embeds" in {i.name for i in decoder.graph.inputs}
        assert "logits" in {o.name for o in decoder.graph.outputs}
        # Vision: pixel_values + pixel_position_ids -> image_features
        vision = pkg["vision_encoder"]
        vision_input_names = {i.name for i in vision.graph.inputs}
        assert "pixel_values" in vision_input_names
        assert "pixel_position_ids" in vision_input_names
        assert "image_features" in {o.name for o in vision.graph.outputs}
        assert component_presence(vision.graph) == "image"
        # Embedding: input_ids + image_features (no audio) -> inputs_embeds
        embedding = pkg["embedding"]
        emb_input_names = {i.name for i in embedding.graph.inputs}
        assert "input_ids" in emb_input_names
        assert "image_features" in emb_input_names
        assert "audio_features" not in emb_input_names
        embedding_image = next(i for i in embedding.graph.inputs if i.name == "image_features")
        assert optional_input_contract(embedding_image) == {
            "presence": "image",
            "absent": {"kind": "zeros", "shape": [0, config.hidden_size]},
        }
        assert "inputs_embeds" in {o.name for o in embedding.graph.outputs}

    def test_gemma4_kv_shared_fallback_attention_is_causal_zero(self):
        """KV-shared layers must use is_causal=0 in the non-GQA fallback.

        The shared-KV fallback feeds the full borrowed K/V as key/value with
        no past (so q_len < kv_len during decode) and relies on the float
        causal bias from ``create_attention_bias`` for masking.  It must NOT
        also set is_causal=1: the ONNX Attention op's built-in causal mask is
        upper-left aligned on CUDA but bottom-right on CPU, so double-masking
        diverges across EPs.  Source/non-shared layers keep is_causal=1.
        """
        from mobius._configs import Gemma4Config

        config = Gemma4Config(
            num_hidden_layers=4,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=16,
            vocab_size=256,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            attn_qk_norm=True,
            layer_types=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
            sliding_window=8,
            global_head_dim=16,
            global_rope_theta=10_000.0,
            global_partial_rotary_factor=0.25,
            final_logit_softcapping=0.0,
            hidden_size_per_layer_input=0,
            pad_token_id=0,
            tie_word_embeddings=True,
            num_kv_shared_layers=2,
        )
        module = registry.get("gemma4_text")(config)
        task = get_task(_default_task_for_model("gemma4_text"))
        pkg = task.build(module, config)
        model = pkg["model"]

        # The fp32 build (no EP) takes the ONNX Attention fallback path.
        is_causal_by_layer: dict[int, int] = {}
        for node in model.graph:
            if node.op_type != "Attention":
                continue
            m = re.search(r"layers\.(\d+)/self_attn", node.name)
            assert m is not None, node.name
            layer_idx = int(m.group(1))
            attr = next(a for a in node.attributes.values() if a.name == "is_causal")
            is_causal_by_layer[layer_idx] = attr.value

        # Source/non-shared layers (0,1) keep is_causal=1; the last
        # num_kv_shared_layers layers (2,3) must use is_causal=0.
        assert is_causal_by_layer == {0: 1, 1: 1, 2: 0, 3: 0}, is_causal_by_layer

    def test_gemma4_moe_graph(self):
        """Build Gemma4 text-only model with enable_moe_block=True (MoE path)."""
        from mobius._configs import Gemma4Config

        config = Gemma4Config(
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=16,
            vocab_size=256,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            attn_qk_norm=True,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=8,
            global_head_dim=16,
            global_rope_theta=10_000.0,
            global_partial_rotary_factor=0.25,
            final_logit_softcapping=0.0,
            hidden_size_per_layer_input=0,
            pad_token_id=0,
            tie_word_embeddings=True,
            # MoE config: every layer has a parallel MoE block
            enable_moe_block=True,
            num_local_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
        )
        model_cls = registry.get("gemma4_text")
        module = model_cls(config)
        task_name = _default_task_for_model("gemma4_text")
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert "model" in pkg
        model = pkg["model"]
        input_names = {i.name for i in model.graph.inputs}
        output_names = {o.name for o in model.graph.outputs}
        assert "input_ids" in input_names
        assert "logits" in output_names

    def test_gemma4_unified_text_graph(self):
        """Build the gemma4_unified (gemma-4-12B) text backbone via Gemma4CausalLMModel.

        ``gemma4_unified_text`` reuses Gemma4CausalLMModel.  This exercises the
        12B-family text architecture: dual head_dim (local 16 / global 32),
        ``attention_k_eq_v`` with a single global KV head, vision-block
        bidirectional attention, and final-logit softcapping.
        """
        from mobius._configs import Gemma4Config

        config = Gemma4Config(
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=256,
            rms_norm_eps=1e-6,
            hidden_act="gelu_pytorch_tanh",
            attn_qk_norm=True,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=8,
            global_head_dim=32,
            global_rope_theta=1_000_000.0,
            global_partial_rotary_factor=0.25,
            final_logit_softcapping=30.0,
            hidden_size_per_layer_input=0,
            num_global_key_value_heads=1,
            attention_k_eq_v=True,
            use_bidirectional_attention="vision",
            pad_token_id=0,
            tie_word_embeddings=True,
        )
        model_cls = registry.get("gemma4_unified_text")
        module = model_cls(config)
        task_name = _default_task_for_model("gemma4_unified_text")
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert "model" in pkg
        model = pkg["model"]
        input_names = {i.name for i in model.graph.inputs}
        output_names = {o.name for o in model.graph.outputs}
        assert "input_ids" in input_names
        assert "logits" in output_names
        # Full-attention layer uses the single global KV head, so its cache
        # entry has a different head_dim than the sliding layer.
        assert "past_key_values.0.key" in input_names
        assert "past_key_values.1.key" in input_names

    def test_gemma4_unified_text_only_emits_gqa(self):
        """text_only build of gemma-4-12B emits GroupQueryAttention on CUDA.

        The multimodal ``gemma4_unified`` decoder uses the bidirectional
        vision-block overlay (float-bias ``Attention``), but the text-only
        export strips ``image_token_id`` / ``use_bidirectional_attention`` so
        the decoder is pure causal and fuses to ``GroupQueryAttention`` on a
        GQA-capable execution provider. This mirrors what
        ``build(text_only=True)`` produces, without network access.
        """
        from collections import Counter

        from mobius._builder import _strip_to_text_only
        from mobius._configs import Gemma4Config
        from mobius.tasks import get_task

        config = Gemma4Config(
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=256,
            rms_norm_eps=1e-6,
            hidden_act="gelu_pytorch_tanh",
            attn_qk_norm=True,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=8,
            global_head_dim=32,
            global_rope_theta=1_000_000.0,
            global_partial_rotary_factor=0.25,
            final_logit_softcapping=30.0,
            hidden_size_per_layer_input=0,
            num_global_key_value_heads=1,
            attention_k_eq_v=True,
            use_bidirectional_attention="vision",
            image_token_id=258880,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
        config = _strip_to_text_only(config, "gemma4_unified_text")
        config.dtype = DTYPE_MAP["f16"]
        assert config.image_token_id is None
        assert config.use_bidirectional_attention is None

        model_cls = registry.get("gemma4_unified_text")
        module = model_cls(config)
        task = get_task(_default_task_for_model("gemma4_unified_text"))
        pkg = build_from_module(module, config, task=task, execution_provider="cuda")

        counts = Counter(n.op_type for n in pkg["model"].graph)
        assert counts.get("GroupQueryAttention", 0) == 2, dict(counts)
        assert counts.get("Attention", 0) == 0, dict(counts)

    def test_strip_to_text_only(self):
        """``_strip_to_text_only`` nulls multimodal fields and sets model_type."""
        from mobius._builder import _strip_to_text_only
        from mobius._configs import Gemma4AudioConfig, Gemma4Config

        config = Gemma4Config(
            model_type="gemma4_unified",
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=256,
            hidden_act="gelu_pytorch_tanh",
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=8,
            global_head_dim=32,
            use_bidirectional_attention="vision",
            image_token_id=258880,
            boa_token_id=256000,
            audio=Gemma4AudioConfig(),
            pad_token_id=0,
        )
        out = _strip_to_text_only(config, "gemma4_unified_text")

        assert out.model_type == "gemma4_unified_text"
        assert out.image_token_id is None
        assert out.use_bidirectional_attention is None
        assert out.boa_token_id is None
        assert out.audio is None
        assert out.vision is None
        # Original config is untouched (dataclasses.replace returns a copy).
        assert config.image_token_id == 258880

    def test_build_text_only_unsupported_model_type_raises(self):
        """``build(text_only=True)`` rejects model types with no text sibling."""
        from unittest import mock

        from mobius._builder import build

        fake_hf = type("HF", (), {"model_type": "llama"})()
        with (
            mock.patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf),
            pytest.raises(ValueError, match="text_only=True is not supported"),
        ):
            build("meta-llama/Llama-3.2-1B", load_weights=False, text_only=True)

    def test_build_text_only_remaps_and_strips(self):
        """``build(text_only=True)`` remaps to the text sibling and strips config.

        Happy path: the model_type is remapped to its text-only registry
        sibling and the multimodal config is stripped before building.
        """
        from unittest import mock

        from mobius import _builder

        fake_hf = type("HF", (), {"model_type": "gemma4_unified"})()
        raw_config = mock.MagicMock(name="raw_config")
        stripped_config = mock.MagicMock(name="stripped_config")
        fake_pkg = mock.MagicMock()
        fake_pkg.items.return_value = []
        fake_module_cls = mock.MagicMock(name="Gemma4CausalLMModel")

        with (
            mock.patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf),
            mock.patch.object(
                _builder.registry, "get", return_value=fake_module_cls
            ) as mock_get,
            mock.patch("mobius._config_resolver._config_from_hf", return_value=raw_config),
            mock.patch(
                "mobius._builder._strip_to_text_only", return_value=stripped_config
            ) as mock_strip,
            mock.patch(
                "mobius._config_resolver._default_task_for_model",
                return_value="text-generation",
            ),
            mock.patch(
                "mobius._builder.build_from_module", return_value=fake_pkg
            ) as mock_build_mod,
        ):
            pkg = _builder.build("google/gemma-4-12B", load_weights=False, text_only=True)

        # model_type was remapped to the text sibling before module lookup
        mock_get.assert_called_once_with("gemma4_unified_text")
        # config stripping invoked with the remapped (text) model_type
        mock_strip.assert_called_once_with(raw_config, "gemma4_unified_text")
        # the stripped config (not the raw multimodal one) is what gets built
        assert mock_build_mod.call_args.args[1] is stripped_config
        assert pkg is fake_pkg

    def test_build_text_only_diffusers_path_raises(self):
        """``build(text_only=True)`` errors on the diffusers/unsupported path.

        When AutoConfig fails and the model is not in the registry, ``build``
        normally falls through to ``build_diffusers_pipeline``. With
        ``text_only=True`` it must raise instead of silently ignoring the flag.
        """
        from unittest import mock

        from mobius._builder import build

        with (
            mock.patch(
                "transformers.AutoConfig.from_pretrained",
                side_effect=ValueError("no such model_type"),
            ),
            mock.patch("mobius._config_resolver._try_load_config_json", return_value=None),
            pytest.raises(ValueError, match="does not resolve to a registered"),
        ):
            build("some/diffusion-pipeline", load_weights=False, text_only=True)

    def test_gemma4_any_to_any_graph(self):
        """Build Gemma4 Any-to-Any model (4-model split: decoder+vision+speech+embedding).

        When ``config.audio`` is set, Gemma4Model adds a ``speech`` model and a
        3-input embedding (input_ids + image_features + audio_features).
        """
        from mobius._configs import Gemma4AudioConfig, Gemma4Config

        config = Gemma4Config(
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=16,
            vocab_size=256,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            attn_qk_norm=True,
            layer_types=["sliding_attention", "sliding_attention"],
            sliding_window=8,
            global_head_dim=16,
            global_rope_theta=10_000.0,
            global_partial_rotary_factor=0.25,
            final_logit_softcapping=0.0,
            hidden_size_per_layer_input=0,
            image_token_id=255999,
            pad_token_id=0,
            tie_word_embeddings=True,
            # num_kv_shared_layers=1 → layer 1 shares KV from layer 0 (same type)
            num_kv_shared_layers=1,
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                patch_size=16,
                norm_eps=1e-6,
            ),
            audio=Gemma4AudioConfig(
                input_size=16,
                hidden_size=32,
                num_layers=1,
                output_dim=64,
                output_proj_dims=64,
                audio_token_id=255998,
            ),
        )
        model_cls = registry.get("gemma4")
        module = model_cls(config)
        task_name = _default_task_for_model("gemma4")
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert set(pkg.keys()) == {
            "decoder",
            "vision_encoder",
            "audio_encoder",
            "embedding",
        }, f"AnyToAny Gemma4 should produce 4 models (with 'audio'), got: {set(pkg.keys())}"
        # Decoder KV cache: num_hidden_layers - num_kv_shared_layers = 1 entry
        decoder = pkg["decoder"]
        decoder_input_names = {i.name for i in decoder.graph.inputs}
        assert "inputs_embeds" in decoder_input_names
        assert "past_key_values.0.key" in decoder_input_names
        assert "past_key_values.1.key" not in decoder_input_names  # shared layer
        assert "logits" in {o.name for o in decoder.graph.outputs}
        # Vision
        vision = pkg["vision_encoder"]
        vision_input_names = {i.name for i in vision.graph.inputs}
        assert "pixel_values" in vision_input_names
        assert "pixel_position_ids" in vision_input_names
        assert "image_features" in {o.name for o in vision.graph.outputs}
        assert component_presence(vision.graph) == "image"
        # Audio encoder
        audio = pkg["audio_encoder"]
        audio_input_names = {i.name for i in audio.graph.inputs}
        assert "input_features" in audio_input_names
        assert "input_features_mask" in audio_input_names
        audio_features = next(o for o in audio.graph.outputs if o.name == "audio_features")
        assert len(audio_features.shape) == 2
        assert audio_features.shape[-1] == config.hidden_size
        assert component_presence(audio.graph) == "audio"
        # Embedding: all three inputs
        embedding = pkg["embedding"]
        emb_input_names = {i.name for i in embedding.graph.inputs}
        assert "input_ids" in emb_input_names
        assert "image_features" in emb_input_names
        assert "audio_features" in emb_input_names
        embedding_image = next(i for i in embedding.graph.inputs if i.name == "image_features")
        assert optional_input_contract(embedding_image) == {
            "presence": "image",
            "absent": {"kind": "zeros", "shape": [0, config.hidden_size]},
        }
        embedding_audio = next(i for i in embedding.graph.inputs if i.name == "audio_features")
        assert optional_input_contract(embedding_audio) == {
            "presence": "audio",
            "absent": {"kind": "zeros", "shape": [0, config.hidden_size]},
        }
        assert "inputs_embeds" in {o.name for o in embedding.graph.outputs}
        # KV cache outputs: num_kv_layers = num_hidden_layers - num_kv_shared_layers = 1
        decoder_output_names = {o.name for o in decoder.graph.outputs}
        assert "present.0.key" in decoder_output_names
        assert "present.0.value" in decoder_output_names
        assert "present.1.key" not in decoder_output_names  # shared layer excluded
        assert "present.1.value" not in decoder_output_names  # shared layer excluded

    @pytest.mark.parametrize(
        ("dtype", "np_dtype"),
        [
            (ir.DataType.FLOAT, np.float32),
            (ir.DataType.FLOAT16, np.float16),
            (ir.DataType.BFLOAT16, ml_dtypes.bfloat16),
        ],
    )
    def test_gemma4_audio_encoder_strips_padding_in_graph(self, dtype, np_dtype):
        """The exported audio graph produces ordered rank-2 valid feature rows."""
        from onnxscript import nn

        from mobius._configs import Gemma4AudioConfig, Gemma4Config
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.tasks._gemma4 import Gemma4Task

        class IdentityAudio(nn.Module):
            def forward(self, op, input_features, input_features_mask=None):
                return op.Identity(input_features), op.Identity(input_features_mask)

        config = Gemma4Config(
            hidden_size=4,
            dtype=dtype,
            audio=Gemma4AudioConfig(input_size=4),
        )
        model = Gemma4Task()._build_audio(IdentityAudio(), config)
        features = np.arange(24, dtype=np_dtype).reshape(2, 3, 4)
        mask = np.array([[True, True, False], [True, False, False]])

        session = OnnxModelSession(model)
        outputs = session.run(
            {
                "input_features": features,
                "input_features_mask": mask,
            }
        )
        session.close()

        np.testing.assert_array_equal(
            outputs["audio_features"],
            np.concatenate([features[0, :2], features[1, :1]], axis=0),
        )
        assert outputs["audio_features"].dtype == np.dtype(np_dtype)

    def test_gemma4_unified_multimodal_graph(self):
        """Build gemma4_unified (gemma-4-12B) encoder-free multimodal model.

        Produces a 4-model split (decoder + vision_encoder + audio_encoder +
        embedding).  The vision/audio encoders are encoder-free embedders
        (no SigLIP/Conformer tower); the decoder uses vision-block
        bidirectional attention, which it derives internally from
        ``input_ids`` (the embedding model does *not* emit
        ``block_sequence_ids``).
        """
        from mobius._configs import Gemma4AudioConfig, Gemma4Config

        config = Gemma4Config(
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=256,
            rms_norm_eps=1e-6,
            hidden_act="gelu_pytorch_tanh",
            attn_qk_norm=True,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=8,
            global_head_dim=32,
            global_rope_theta=1_000_000.0,
            global_partial_rotary_factor=0.25,
            final_logit_softcapping=30.0,
            hidden_size_per_layer_input=0,
            num_global_key_value_heads=1,
            attention_k_eq_v=True,
            use_bidirectional_attention="vision",
            image_token_id=255999,
            pad_token_id=0,
            tie_word_embeddings=True,
            vision=VisionConfig(
                hidden_size=48,
                patch_size=4,
                pooling_kernel_size=2,
                position_embedding_size=64,
                out_hidden_size=48,
                norm_eps=1e-6,
            ),
            audio=Gemma4AudioConfig(
                hidden_size=40,
                output_proj_dims=40,
                audio_token_id=255998,
            ),
        )
        model_cls = registry.get("gemma4_unified")
        module = model_cls(config)
        task = get_task(_default_task_for_model("gemma4_unified"))
        pkg = build_from_module(module, config, task=task)

        assert set(pkg.keys()) == {
            "decoder",
            "vision_encoder",
            "audio_encoder",
            "embedding",
        }, f"gemma4_unified should produce 4 models, got: {set(pkg.keys())}"

        # Vision embedder: raw patches (no encoder layers) → image_features
        vision = pkg["vision_encoder"]
        v_inputs = {i.name for i in vision.graph.inputs}
        assert v_inputs == {"pixel_values", "pixel_position_ids"}
        assert "image_features" in {o.name for o in vision.graph.outputs}
        assert component_presence(vision.graph) == "image"

        # Audio embedder: raw frames + mask → audio_features
        audio = pkg["audio_encoder"]
        a_inputs = {i.name for i in audio.graph.inputs}
        assert a_inputs == {"input_features", "input_features_mask"}
        assert "audio_features" in {o.name for o in audio.graph.outputs}
        assert component_presence(audio.graph) == "audio"

        # Embedding: fuses both modalities → inputs_embeds (no block_sequence_ids;
        # the decoder derives the bidirectional overlay from input_ids itself)
        embedding = pkg["embedding"]
        e_inputs = {i.name for i in embedding.graph.inputs}
        assert {"input_ids", "image_features", "audio_features"} <= e_inputs
        embedding_image = next(i for i in embedding.graph.inputs if i.name == "image_features")
        assert optional_input_contract(embedding_image) == {
            "presence": "image",
            "absent": {"kind": "zeros", "shape": [0, config.hidden_size]},
        }
        embedding_audio = next(i for i in embedding.graph.inputs if i.name == "audio_features")
        assert optional_input_contract(embedding_audio) == {
            "presence": "audio",
            "absent": {"kind": "zeros", "shape": [0, config.hidden_size]},
        }
        e_outputs = {o.name for o in embedding.graph.outputs}
        assert "inputs_embeds" in e_outputs
        assert "block_sequence_ids" not in e_outputs

        # Decoder: consumes inputs_embeds + input_ids (for the vision-block
        # bidirectional overlay, derived internally)
        decoder = pkg["decoder"]
        d_inputs = {i.name for i in decoder.graph.inputs}
        assert "inputs_embeds" in d_inputs
        assert "input_ids" in d_inputs
        assert "block_sequence_ids" not in d_inputs
        assert "logits" in {o.name for o in decoder.graph.outputs}

    def test_gemma4_kv_shared_layer_tracing(self):
        """Verify all num_hidden_layers are traced and KV outputs = num_kv_layers.

        With num_kv_shared_layers=1 and num_hidden_layers=2:
        - Both layers must be traced (Attention op count = 2)
        - KV cache inputs: 1 entry (only layer 0 has its own K/V)
        - KV cache outputs: 1 entry (shared layer excluded from present_key_values)
        """
        from mobius._configs import Gemma4AudioConfig, Gemma4Config
        from mobius.tasks._gemma4 import Gemma4Task

        config = Gemma4Config(
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=16,
            vocab_size=256,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            attn_qk_norm=True,
            layer_types=["sliding_attention", "sliding_attention"],
            sliding_window=8,
            global_head_dim=16,
            global_rope_theta=10_000.0,
            global_partial_rotary_factor=0.25,
            final_logit_softcapping=0.0,
            hidden_size_per_layer_input=0,
            image_token_id=255999,
            pad_token_id=0,
            tie_word_embeddings=True,
            num_kv_shared_layers=1,
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                patch_size=16,
                norm_eps=1e-6,
            ),
            audio=Gemma4AudioConfig(
                input_size=16,
                hidden_size=32,
                num_layers=1,
                output_dim=64,
                output_proj_dims=64,
                audio_token_id=255998,
            ),
        )
        model_cls = registry.get("gemma4")
        module = model_cls(config)
        task = Gemma4Task()
        pkg = task.build(module, config)
        decoder = pkg["decoder"]

        # All num_hidden_layers=2 layers must be traced: each produces one Attention op.
        attention_nodes = [n for n in decoder.graph if n.op_type == "Attention"]
        assert len(attention_nodes) == config.num_hidden_layers, (
            f"Expected {config.num_hidden_layers} Attention ops (all layers traced), "
            f"got {len(attention_nodes)}"
        )

        # KV cache inputs: exactly num_kv_layers = 1 (shared layer has no own KV)
        input_names = {i.name for i in decoder.graph.inputs}
        assert "past_key_values.0.key" in input_names
        assert "past_key_values.1.key" not in input_names

        # KV cache outputs: exactly num_kv_layers = 1 (shared layer excluded)
        output_names = {o.name for o in decoder.graph.outputs}
        assert "present.0.key" in output_names
        assert "present.0.value" in output_names
        assert "present.1.key" not in output_names
        assert "present.1.value" not in output_names

    def test_gemma4_k_eq_v_with_global_kv_heads(self):
        """Verify attention_k_eq_v removes v_proj and num_global_key_value_heads sets KV cache shapes.

        Config: attention_k_eq_v=True, num_key_value_heads=4 (sliding),
        num_global_key_value_heads=2 (full). Full-attention layers should:
        - Have no v_proj initializer (V=K)
        - Use num_global_key_value_heads=2 for KV cache shapes
        Sliding layers should use num_key_value_heads=4.
        """
        from mobius._configs import Gemma4Config
        from mobius.models.gemma4 import Gemma4CausalLMModel
        from mobius.tasks._gemma4 import Gemma4TextCausalLMTask

        config = Gemma4Config(
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            vocab_size=256,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            attn_qk_norm=True,
            # Layer 0: sliding, Layer 1: full (k_eq_v + global heads)
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=8,
            global_head_dim=16,
            global_rope_theta=10_000.0,
            global_partial_rotary_factor=0.25,
            final_logit_softcapping=0.0,
            hidden_size_per_layer_input=0,
            image_token_id=255999,
            pad_token_id=0,
            tie_word_embeddings=True,
            attention_k_eq_v=True,
            num_global_key_value_heads=2,
        )
        module = Gemma4CausalLMModel(config)
        task = Gemma4TextCausalLMTask()
        pkg = task.build(module, config)
        decoder = pkg["model"]

        # Check initializer names: full-attention layer (1) should have no v_proj
        init_names = set(decoder.graph.initializers)
        # Sliding layer 0 has k_proj, v_proj
        assert "model.layers.0.self_attn.k_proj.weight" in init_names
        assert "model.layers.0.self_attn.v_proj.weight" in init_names
        # Full layer 1 has k_proj but NO v_proj (k_eq_v: V=K)
        assert "model.layers.1.self_attn.k_proj.weight" in init_names
        assert "model.layers.1.self_attn.v_proj.weight" not in init_names

        # KV cache shapes:
        # Layer 0 (sliding): num_key_value_heads=4
        # Layer 1 (full): num_global_key_value_heads=2
        input_shapes = {i.name: list(i.shape) for i in decoder.graph.inputs}
        # Layer 0: kv_heads=4
        layer0_key_shape = input_shapes["past_key_values.0.key"]
        assert layer0_key_shape[1] == 4, (
            f"Sliding layer 0 should have 4 KV heads, got {layer0_key_shape[1]}"
        )
        # Layer 1: kv_heads=2 (num_global_key_value_heads)
        layer1_key_shape = input_shapes["past_key_values.1.key"]
        assert layer1_key_shape[1] == 2, (
            f"Full layer 1 should have 2 KV heads "
            f"(num_global_key_value_heads), got {layer1_key_shape[1]}"
        )

    def test_gemma3n_multimodal_graph(self):
        """Build Gemma 3n via the registry (4-model split, audio configured).

        The tiny config is taken from ``VL_CONFIGS`` so this test and the
        parametrized suite can never drift apart.
        """
        config = _base_config(**vl_overrides("gemma3n"))
        module = registry.get("gemma3n")(config)
        task_name = _default_task_for_model("gemma3n")
        assert task_name == "gemma3n"
        pkg = get_task(task_name).build(module, config)

        assert set(pkg.keys()) == {
            "decoder",
            "vision_encoder",
            "audio_encoder",
            "embedding",
        }, f"Gemma3n with audio should produce 4 models, got: {set(pkg.keys())}"

        # --- decoder: inputs_embeds + per_layer_inputs -> logits + KV cache.
        # The per-layer embedding tables live in the embedding sub-model, so
        # the decoder takes their combined output as a graph input and never
        # sees input_ids.
        decoder = pkg["decoder"]
        decoder_inputs = {i.name: i for i in decoder.graph.inputs}
        assert "input_ids" not in decoder_inputs
        assert "inputs_embeds" in decoder_inputs
        assert "per_layer_inputs" in decoder_inputs
        assert list(decoder_inputs["per_layer_inputs"].shape)[-1] == (
            config.num_hidden_layers * config.hidden_size_per_layer_input
        )
        decoder_outputs = {o.name for o in decoder.graph.outputs}
        assert "logits" in decoder_outputs
        # num_kv_shared_layers=0 here, so every layer owns a cache entry.
        assert module.decoder.kv_cache_layer_count() == config.num_hidden_layers
        for i in range(config.num_hidden_layers):
            assert f"past_key_values.{i}.key" in decoder_inputs
            assert f"present.{i}.key" in decoder_outputs

        # --- vision: fixed-size pixels -> [B*256, hidden], no mask.
        vision = pkg["vision_encoder"]
        vision_inputs = {i.name: i for i in vision.graph.inputs}
        assert set(vision_inputs) == {"pixel_values"}
        image_size = config.vision.image_size
        assert list(vision_inputs["pixel_values"].shape)[1:] == [3, image_size, image_size]
        image_features = next(o for o in vision.graph.outputs if o.name == "image_features")
        assert len(image_features.shape) == 2
        assert image_features.shape[-1] == config.hidden_size
        assert component_presence(vision.graph) == "image"

        # --- audio: mel frames + bool mask -> fixed-count [B*188, hidden].
        # Unlike Gemma 4, padded rows are not stripped, so there is no
        # companion audio_features_mask output.
        audio = pkg["audio_encoder"]
        audio_inputs = {i.name: i for i in audio.graph.inputs}
        assert set(audio_inputs) == {"input_features", "input_features_mask"}
        assert list(audio_inputs["input_features"].shape)[-1] == (config.audio.input_feat_size)
        assert audio_inputs["input_features_mask"].dtype == ir.DataType.BOOL
        assert {o.name for o in audio.graph.outputs} == {"audio_features"}
        audio_features = next(o for o in audio.graph.outputs if o.name == "audio_features")
        assert len(audio_features.shape) == 2
        assert audio_features.shape[-1] == config.hidden_size
        assert component_presence(audio.graph) == "audio"

        # --- embedding: ids + both feature sets -> inputs_embeds + per_layer.
        embedding = pkg["embedding"]
        emb_inputs = {i.name: i for i in embedding.graph.inputs}
        assert set(emb_inputs) == {"input_ids", "image_features", "audio_features"}
        for name, presence in (("image_features", "image"), ("audio_features", "audio")):
            assert optional_input_contract(emb_inputs[name]) == {
                "presence": presence,
                "absent": {"kind": "zeros", "shape": [0, config.hidden_size]},
            }, name
        assert {o.name for o in embedding.graph.outputs} == {
            "inputs_embeds",
            "per_layer_inputs",
        }
        # The embedding's per_layer_inputs must match what the decoder expects.
        emb_per_layer = next(
            o for o in embedding.graph.outputs if o.name == "per_layer_inputs"
        )
        assert (
            list(emb_per_layer.shape)[-1]
            == (list(decoder_inputs["per_layer_inputs"].shape)[-1])
        )
        # The 4.7 GB per-layer table belongs to the embedding model only.
        assert "embedding.embed_tokens_per_layer.weight" in embedding.graph.initializers
        assert not any("embed_tokens_per_layer" in n for n in decoder.graph.initializers)
        # Only the *hard* embedder path is built here, so the soft-path norm
        # (which the towers own) must not add a dangling initializer.
        assert "embedding.embed_vision.hard_embedding_norm.weight" in (
            embedding.graph.initializers
        )
        assert not any("soft_embedding_norm" in n for n in embedding.graph.initializers)

    def test_gemma3n_multimodal_graph_without_audio(self):
        """With ``config.audio=None`` the package drops the audio encoder.

        The embedding model must then also drop its ``audio_features`` input,
        or the runtime would be asked for a tensor no component produces.
        """
        overrides = vl_overrides("gemma3n")
        overrides["audio"] = None
        overrides["audio_token_id"] = None
        config = _base_config(**overrides)
        module = registry.get("gemma3n")(config)
        pkg = get_task("gemma3n").build(module, config)

        assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}
        assert module.audio_encoder is None
        emb_input_names = {i.name for i in pkg["embedding"].graph.inputs}
        assert "image_features" in emb_input_names
        assert "audio_features" not in emb_input_names
        # No audio embedder weights should be built either.
        assert not any("embed_audio" in n for n in pkg["embedding"].graph.initializers)

    def test_blip2_vision_language_graph(self):
        """Build BLIP-2 with ViT + Q-Former + LLM 3-model split."""
        config = _base_config(
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
                norm_eps=1e-6,
            ),
            image_token_id=50265,
            # Q-Former config
            num_query_tokens=4,
            qformer_hidden_size=32,
            qformer_num_hidden_layers=1,
            qformer_num_attention_heads=2,
            qformer_intermediate_size=64,
        )
        model_cls = registry.get("blip-2")
        module = model_cls(config)
        task_name = _default_task_for_model("blip-2")
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}

        # Decoder: inputs_embeds → logits + KV cache
        decoder = pkg["decoder"]
        assert "inputs_embeds" in {i.name for i in decoder.graph.inputs}
        assert "logits" in {o.name for o in decoder.graph.outputs}

        # Vision: pixel_values → image_features (via ViT + Q-Former)
        vision = pkg["vision_encoder"]
        assert "pixel_values" in {i.name for i in vision.graph.inputs}
        assert "image_features" in {o.name for o in vision.graph.outputs}

        # Embedding: input_ids + image_features → inputs_embeds
        embed = pkg["embedding"]
        assert "input_ids" in {i.name for i in embed.graph.inputs}
        assert "inputs_embeds" in {o.name for o in embed.graph.outputs}

    def test_llava_aliases_build(self):
        """LLaVA aliases (llava_next, llava_onevision, video_llava, etc.) all build."""
        config = _base_config(
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
                norm_eps=1e-6,
            ),
            image_token_id=32000,
        )
        for model_type in (
            "aya_vision",
            "cohere2_vision",
            "deepseek_vl",
            "deepseek_vl_hybrid",
            "glm4v",
            "glm4v_moe",
            "got_ocr2",
            "instructblipvideo",
            "janus",
            "llava_next",
            "llava_next_video",
            "llava_onevision",
            "ovis2",
            "smolvlm",
            "video_llava",
            "vipllava",
            "chameleon",
            "florence2",
            "fuyu",
            "idefics2",
            "idefics3",
            "instructblip",
            "molmo",
            "paligemma",
            "pixtral",
        ):
            model_cls = registry.get(model_type)
            module = model_cls(config)
            task_name = _default_task_for_model(model_type)
            task = get_task(task_name)
            pkg = task.build(module, config)

            assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}, (
                f"{model_type} should produce 3 models"
            )
            assert "logits" in {o.name for o in pkg["decoder"].graph.outputs}, (
                f"{model_type} decoder missing logits"
            )
            assert "pixel_values" in {i.name for i in pkg["vision_encoder"].graph.inputs}, (
                f"{model_type} vision missing pixel_values"
            )

    def test_mistral3_pixtral_vision_build(self):
        """Build Mistral-3 with Pixtral vision encoder (2D RoPE + patch merge)."""
        config = _base_config(
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
                norm_eps=1e-6,
                model_type="pixtral",
            ),
            image_token_id=32000,
        )
        model_cls = registry.get("mistral3")
        module = model_cls(config)
        task_name = _default_task_for_model("mistral3")
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}
        assert "logits" in {o.name for o in pkg["decoder"].graph.outputs}
        assert "pixel_values" in {i.name for i in pkg["vision_encoder"].graph.inputs}

    def test_pixtral_preprocess_weights_remapping(self):
        """Verify _preprocess_pixtral_weights remaps HF weight names correctly."""
        import torch

        from mobius.models.llava import _preprocess_pixtral_weights

        state_dict = {
            "vision_tower.patch_conv.weight": torch.zeros(1),
            "multi_modal_projector.norm.weight": torch.zeros(1),
            "language_model.model.embed_tokens.weight": torch.zeros(1),
            "language_model.lm_head.weight": torch.zeros(1),
            "language_model.model.layers.0.self_attn.q_proj.weight": torch.zeros(1),
        }
        result = _preprocess_pixtral_weights(state_dict, tie_word_embeddings=False)

        # Vision/projector keys get vision_encoder. prefix
        assert "vision_encoder.vision_tower.patch_conv.weight" in result
        assert "vision_encoder.multi_modal_projector.norm.weight" in result
        # embed_tokens duplicated to decoder and embedding
        assert "decoder.model.embed_tokens.weight" in result
        assert "embedding.embed_tokens.weight" in result
        # lm_head remapped under decoder
        assert "decoder.lm_head.weight" in result
        # Other language_model keys remapped under decoder
        assert "decoder.model.layers.0.self_attn.q_proj.weight" in result
        # Original keys should not be present
        for original_key in state_dict:
            assert original_key not in result

        # tie_word_embeddings=True creates decoder.lm_head.weight from embed_tokens
        state_dict_tied = {
            "language_model.model.embed_tokens.weight": torch.zeros(1),
        }
        result_tied = _preprocess_pixtral_weights(state_dict_tied, tie_word_embeddings=True)
        assert "decoder.lm_head.weight" in result_tied
        assert "decoder.model.embed_tokens.weight" in result_tied
        assert "embedding.embed_tokens.weight" in result_tied

    def test_pixtral_preprocess_weights_model_prefix_strip(self):
        """Verify outer model. prefix is stripped (Mistral3ForConditionalGeneration)."""
        import torch

        from mobius.models.llava import _preprocess_pixtral_weights

        state_dict = {
            "model.vision_tower.patch_conv.weight": torch.zeros(1),
            "model.language_model.model.layers.0.mlp.gate_proj.weight": torch.zeros(1),
            "lm_head.weight": torch.zeros(1),
        }
        result = _preprocess_pixtral_weights(state_dict, tie_word_embeddings=False)

        # model. prefix stripped, then vision gets vision_encoder. prefix
        assert "vision_encoder.vision_tower.patch_conv.weight" in result
        # model. prefix stripped, then language_model remapped to decoder
        assert "decoder.model.layers.0.mlp.gate_proj.weight" in result
        # bare lm_head gets decoder. prefix
        assert "decoder.lm_head.weight" in result

    def test_mllama_vision_language_graph(self):
        """Build Mllama (Llama 3.2 Vision) with cross-attention decoder."""
        from mobius._configs import MllamaConfig

        config = _base_config(
            config_cls=MllamaConfig,
            num_hidden_layers=3,
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
                norm_eps=1e-6,
            ),
            image_token_id=32000,
            cross_attention_layers=[1],
        )
        model_cls = registry.get("mllama")
        module = model_cls(config)
        task_name = _default_task_for_model("mllama")
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}

        decoder = pkg["decoder"]
        dec_inputs = {i.name for i in decoder.graph.inputs}
        assert "inputs_embeds" in dec_inputs
        assert "logits" in {o.name for o in decoder.graph.outputs}

        # Cross-attention states must be a decoder input
        assert "cross_attention_states" in dec_inputs

        # Cross-attention layers (layer 1) should use a different
        # past-sequence-length dim than self-attention layers (0, 2)
        kv_shapes = {}
        for inp in decoder.graph.inputs:
            if inp.name.startswith("past_key_values."):
                kv_shapes[inp.name] = str(inp.shape)
        assert kv_shapes["past_key_values.1.key"] != kv_shapes["past_key_values.0.key"]
        assert kv_shapes["past_key_values.0.key"] == kv_shapes["past_key_values.2.key"]

        vision = pkg["vision_encoder"]
        assert "pixel_values" in {i.name for i in vision.graph.inputs}

        embed = pkg["embedding"]
        assert "input_ids" in {i.name for i in embed.graph.inputs}
        assert "inputs_embeds" in {o.name for o in embed.graph.outputs}

    def test_deepseek_ocr2_graph(self):
        """Build DeepSeek-OCR-2 with 3-model VL split."""
        config = _base_config(
            # LLM decoder: DeepSeek-V2 non-MLA + MoE
            qk_nope_head_dim=0,
            qk_rope_head_dim=0,
            v_head_dim=0,
            num_local_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=1.0,
            scoring_func="softmax",
            topk_method="greedy",
            first_k_dense_replace=1,
            n_shared_experts=2,
            image_token_id=100015,
        )
        model_cls = registry.get("deepseek_vl_v2")
        module = model_cls(config)
        task_name = _default_task_for_model("deepseek_vl_v2")
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}

        decoder = pkg["decoder"]
        assert "inputs_embeds" in {i.name for i in decoder.graph.inputs}
        assert "logits" in {o.name for o in decoder.graph.outputs}

        vision = pkg["vision_encoder"]
        assert "pixel_values" in {i.name for i in vision.graph.inputs}
        assert "image_features" in {o.name for o in vision.graph.outputs}

        embed = pkg["embedding"]
        assert "input_ids" in {i.name for i in embed.graph.inputs}
        assert "inputs_embeds" in {o.name for o in embed.graph.outputs}

    def test_vl_aliases_resolve(self):
        """Verify VL alias model_types resolve to the same class and task."""
        from mobius.models.qwen35 import (
            Qwen35VL3ModelCausalLMModel,
        )
        from mobius.models.qwen_vl import (
            Qwen2VLCausalLMModel,
            Qwen25VLCausalLMModel,
            Qwen25VLTextModel,
        )

        # Qwen2-VL has its own model class (LayerNorm + FCMLP vision)
        assert registry.get("qwen2_vl") is Qwen2VLCausalLMModel
        assert _default_task_for_model("qwen2_vl") == "qwen-vl"

        # Qwen2.5-VL is separate (RMSNorm + GatedMLP vision)
        assert registry.get("qwen2_5_vl") is Qwen25VLCausalLMModel

        assert registry.get("qwen2_vl_text") is Qwen25VLTextModel
        assert registry.get("qwen2_vl_text") is registry.get("qwen2_5_vl_text")

        assert registry.get("qwen3_5") is Qwen35VL3ModelCausalLMModel
        assert registry.get("qwen3_5") is registry.get("qwen3_5_vl")
        assert _default_task_for_model("qwen3_5") == "hybrid-qwen-vl"

    def test_qwen35_vl_preprocess_weights_model_prefix(self):
        """Qwen3.5-VL preprocess handles model.language_model.* style keys."""
        import torch

        from mobius.models.qwen35 import Qwen35VL3ModelCausalLMModel

        vision_config = VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            patch_size=16,
            in_channels=3,
            out_hidden_size=64,
            num_position_embeddings=16,
        )
        config = _base_config(vision=vision_config)
        module = Qwen35VL3ModelCausalLMModel(config)
        embed_weight = torch.zeros(config.vocab_size, config.hidden_size)
        state_dict = {
            "model.language_model.embed_tokens.weight": embed_weight,
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(
                config.hidden_size, config.hidden_size
            ),
            "model.language_model.lm_head.weight": torch.zeros(
                config.vocab_size, config.hidden_size
            ),
            "model.visual.blocks.0.mlp.linear_fc1.weight": torch.zeros(
                vision_config.intermediate_size, vision_config.hidden_size
            ),
            "mtp_head.weight": torch.zeros(1),
        }

        result = module.preprocess_weights(state_dict)

        assert "decoder.model.embed_tokens.weight" in result
        assert "embedding.embed_tokens.weight" in result
        assert (
            result["decoder.model.embed_tokens.weight"]
            is result["embedding.embed_tokens.weight"]
        )
        assert "decoder.model.layers.0.self_attn.q_proj.weight" in result
        assert "decoder.lm_head.weight" in result
        assert "vision_encoder.visual.blocks.0.mlp.up_proj.weight" in result
        assert "mtp_head.weight" not in result


class TestBuildGraphDtype:
    """Verify dtype casting for model initializers."""

    @pytest.mark.parametrize(
        "dtype_str,expected",
        [
            ("f16", "FLOAT16"),
            ("bf16", "BFLOAT16"),
        ],
    )
    def test_dtype_casts_float_initializers(self, dtype_str, expected):
        """Build with dtype and verify Parameter-derived initializers are cast."""
        config = _base_config()
        config.dtype = DTYPE_MAP[dtype_str]
        model_cls = registry.get("llama")
        module = model_cls(config)
        model = build_from_module(module, config)["model"]

        expected_dtype = ir.DataType[expected]
        for name, init in model.graph.initializers.items():
            if init.dtype == ir.DataType.INT64:
                continue
            # Lifted scalar constants (e.g. const_1.0_f32) stay f32
            if name.startswith("const_"):
                continue
            assert init.dtype == expected_dtype, (
                f"Initializer '{name}' dtype is {init.dtype}, expected {expected_dtype}"
            )

    @pytest.mark.parametrize(
        "dtype_str",
        ["f16", "bf16"],
    )
    def test_multimodal_encoder_inputs_match_model_dtype(self, dtype_str):
        """Vision/audio encoder graph inputs use the model's compute dtype.

        ORT GenAI's image/audio processor output doesn't have to be f32 —
        the runtime can deliver inputs at the model's compute dtype
        directly. Declaring the graph input as ``config.dtype`` removes
        an extra Cast node at graph entry.
        """
        # Use a VL model with 3-model split (vision_encoder is separate)
        config = _base_config(
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
                norm_eps=1e-6,
            ),
            image_token_id=32000,
        )
        config.dtype = DTYPE_MAP[dtype_str]
        model_cls = registry.get("llava")
        module = model_cls(config)
        task = get_task("vision-language")
        pkg = task.build(module, config)

        # Vision encoder pixel_values input must match config.dtype
        vision_model = pkg["vision_encoder"]
        pixel_values_input = vision_model.graph.inputs[0]
        assert pixel_values_input.name == "pixel_values"
        assert pixel_values_input.dtype == config.dtype, (
            f"Vision encoder input dtype is {pixel_values_input.dtype}, "
            f"expected {config.dtype} (no Cast at graph entry)"
        )

        # No Cast as the first node — inputs already arrive at the right dtype
        first_node = next(iter(vision_model.graph))
        assert first_node.op_type != "Cast" or first_node.inputs[0].name != "pixel_values", (
            f"Unexpected Cast on pixel_values: graph input dtype {pixel_values_input.dtype} "
            f"should already match the encoder's compute dtype."
        )

    @pytest.mark.parametrize(
        "dtype_str",
        ["f16", "bf16"],
    )
    def test_gemma4_encoder_inputs_match_model_dtype(self, dtype_str):
        """Gemma4 vision/audio encoder inputs match the model's compute dtype."""
        from mobius._configs import Gemma4AudioConfig, Gemma4Config

        config = Gemma4Config(
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=16,
            vocab_size=256,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            attn_qk_norm=True,
            layer_types=["sliding_attention", "sliding_attention"],
            sliding_window=8,
            global_head_dim=16,
            global_rope_theta=10_000.0,
            global_partial_rotary_factor=0.25,
            final_logit_softcapping=0.0,
            hidden_size_per_layer_input=0,
            image_token_id=255999,
            pad_token_id=0,
            tie_word_embeddings=True,
            num_kv_shared_layers=1,
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                patch_size=16,
                norm_eps=1e-6,
            ),
            audio=Gemma4AudioConfig(
                input_size=16,
                hidden_size=32,
                num_layers=1,
                output_dim=64,
                output_proj_dims=64,
                audio_token_id=255998,
            ),
            dtype=DTYPE_MAP[dtype_str],
        )
        model_cls = registry.get("gemma4")
        module = model_cls(config)
        task = get_task("gemma4")
        pkg = task.build(module, config)

        # Vision encoder pixel_values must match config.dtype
        vision_model = pkg["vision_encoder"]
        pv_input = vision_model.graph.inputs[0]
        assert pv_input.name == "pixel_values"
        assert pv_input.dtype == config.dtype

        # Audio encoder input_features must match config.dtype
        audio_model = pkg["audio_encoder"]
        af_input = audio_model.graph.inputs[0]
        assert af_input.name == "input_features"
        assert af_input.dtype == config.dtype


class TestBuildGraphMultiModal:
    """Verify Phi4MM builds with Phi4MMMultiModalTask (4-model split)."""

    def test_phi4mm_multimodal_graph(self):
        """Build Phi4MM 4-model split and verify all components."""
        config = _base_config(
            partial_rotary_factor=0.5,
            rope_type="longrope",
            rope_scaling={
                "short_factor": LONGROPE_FACTORS,
                "long_factor": LONGROPE_FACTORS,
            },
            original_max_position_embeddings=128,
            vision=VisionConfig(
                lora={"r": 4, "lora_alpha": 8},
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
                norm_eps=1e-6,
            ),
            audio=AudioConfig(
                lora={"r": 8, "lora_alpha": 16},
                attention_dim=32,
                attention_heads=2,
                num_blocks=1,
                linear_units=64,
                kernel_size=3,
                input_size=16,
                conv_channels=32,
                t5_bias_max_distance=10,
                token_id=200011,
            ),
            image_token_id=200010,
        )
        model_cls = registry.get("phi4mm")
        module = model_cls(config)
        task = Phi4MMMultiModalTask()
        pkg = task.build(module, config)

        # Verify 4 models in package
        assert len(pkg) == 4, f"Expected 4 models, got {len(pkg)}"
        for key in ("vision_encoder", "audio_encoder", "embedding", "decoder"):
            assert key in pkg, f"Missing model: {key}"

        # Vision model has SigLIP encoder initializers
        vision_inits = list(pkg["vision_encoder"].graph.initializers)
        assert any("img_processor" in n for n in vision_inits), (
            "Vision model should have SigLIP initializers"
        )

        # Speech model has Conformer encoder initializers
        speech_inits = list(pkg["audio_encoder"].graph.initializers)
        assert any("encoder" in n for n in speech_inits), (
            "Speech model should have Conformer initializers"
        )

        # Decoder model (pkg["decoder"]) has LoRA initializers
        decoder_inits = list(pkg["decoder"].graph.initializers)
        assert any("lora" in n for n in decoder_inits), (
            "Decoder model should have LoRA initializers"
        )

    def test_phi4_multimodal_alias_resolves(self):
        """Verify phi4_multimodal alias resolves to same class as phi4mm."""
        from mobius.models.phi import Phi4MMMultiModalModel

        assert registry.get("phi4_multimodal") is Phi4MMMultiModalModel
        assert registry.get("phi4_multimodal") is registry.get("phi4mm")
        assert _default_task_for_model("phi4_multimodal") == "phi4mm-multimodal"

    def test_phi3_v_vision_language_graph(self):
        """Build Phi-3-Vision with 3-model split and verify all components."""
        config = _base_config(
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
                norm_eps=1e-5,
            ),
            image_token_id=32044,
        )
        model_cls = registry.get("phi3_v")
        module = model_cls(config)
        task_name = _default_task_for_model("phi3_v")
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}

        decoder = pkg["decoder"]
        assert "inputs_embeds" in {i.name for i in decoder.graph.inputs}
        assert "logits" in {o.name for o in decoder.graph.outputs}

        vision = pkg["vision_encoder"]
        assert "pixel_values" in {i.name for i in vision.graph.inputs}
        assert "image_features" in {o.name for o in vision.graph.outputs}

        embed = pkg["embedding"]
        assert "input_ids" in {i.name for i in embed.graph.inputs}
        assert "inputs_embeds" in {o.name for o in embed.graph.outputs}

    def test_phi4_siglip_vision_language_graph(self):
        """Build Phi-4-Reasoning-Vision (phi4-siglip) with 3-model split and verify components."""
        config = _base_config(
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
                norm_eps=1e-6,
            ),
            image_token_id=-200,
        )
        model_cls = registry.get("phi4-siglip")
        module = model_cls(config)
        task_name = _default_task_for_model("phi4-siglip")
        task = get_task(task_name)
        pkg = task.build(module, config)

        assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}

        decoder = pkg["decoder"]
        assert "inputs_embeds" in {i.name for i in decoder.graph.inputs}
        assert "logits" in {o.name for o in decoder.graph.outputs}

        vision = pkg["vision_encoder"]
        assert "pixel_values" in {i.name for i in vision.graph.inputs}
        assert "image_features" in {o.name for o in vision.graph.outputs}

        embed = pkg["embedding"]
        assert "input_ids" in {i.name for i in embed.graph.inputs}
        assert "inputs_embeds" in {o.name for o in embed.graph.outputs}

    def test_phi3_v_decoder_excludes_vision_weights(self):
        """Decoder weight preprocessing must not retain vision-only checkpoint tensors."""
        import torch

        from mobius.models.phi3_v import _Phi3VDecoderModel

        config = _base_config(
            vision=VisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=28,
                patch_size=14,
            ),
            image_token_id=32044,
        )
        weights = {
            "model.layers.0.self_attn.qkv_proj.weight": torch.zeros(128, 64),
            "model.vision_embed_tokens.img_processor.vision_model.weight": torch.zeros(1),
            "lm_head.weight": torch.zeros(256, 64),
        }
        remapped = _Phi3VDecoderModel(config).preprocess_weights(weights)

        assert "model.vision_embed_tokens.img_processor.vision_model.weight" not in remapped
        assert "lm_head.weight" in remapped


class TestBuildGraphWhisper:
    """Verify Whisper encoder-decoder builds with SpeechToTextTask."""

    def _whisper_config(self):
        from mobius._configs import WhisperConfig

        return WhisperConfig(
            vocab_size=512,
            hidden_size=TINY_HIDDEN,
            intermediate_size=TINY_INTERMEDIATE,
            num_hidden_layers=TINY_LAYERS,
            num_attention_heads=TINY_HEADS,
            num_key_value_heads=TINY_HEADS,
            head_dim=TINY_HIDDEN // TINY_HEADS,
            hidden_act="gelu",
            pad_token_id=0,
            tie_word_embeddings=True,
            attn_qkv_bias=True,
            attn_o_bias=True,
            encoder_layers=TINY_LAYERS,
            encoder_attention_heads=TINY_HEADS,
            encoder_ffn_dim=TINY_INTERMEDIATE,
            num_mel_bins=16,
            max_source_positions=100,
            max_target_positions=50,
            scale_embedding=True,
        )

    def test_whisper_package_builds(self):
        """Build Whisper with SpeechToTextTask and verify encoder + decoder."""
        from mobius._builder import build_from_module
        from mobius.models.whisper import WhisperForConditionalGeneration
        from mobius.tasks import SpeechToTextTask

        config = self._whisper_config()
        module = WhisperForConditionalGeneration(config)
        task = SpeechToTextTask()
        pkg = build_from_module(module, config, task=task)

        assert "encoder" in pkg
        assert "decoder" in pkg

    def test_whisper_encoder_io(self):
        """Verify encoder inputs/outputs."""
        from mobius._builder import build_from_module
        from mobius.models.whisper import WhisperForConditionalGeneration
        from mobius.tasks import SpeechToTextTask

        config = self._whisper_config()
        module = WhisperForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=SpeechToTextTask())
        encoder = pkg["encoder"]

        input_names = {inp.name for inp in encoder.graph.inputs}
        output_names = {out.name for out in encoder.graph.outputs}
        assert "input_features" in input_names
        assert "encoder_hidden_states" in output_names

    def test_whisper_decoder_io(self):
        """Verify decoder inputs/outputs including KV cache."""
        from mobius._builder import build_from_module
        from mobius.models.whisper import WhisperForConditionalGeneration
        from mobius.tasks import SpeechToTextTask

        config = self._whisper_config()
        module = WhisperForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=SpeechToTextTask())
        decoder = pkg["decoder"]

        input_names = {inp.name for inp in decoder.graph.inputs}
        output_names = {out.name for out in decoder.graph.outputs}

        assert "decoder_input_ids" in input_names
        assert "encoder_hidden_states" in input_names
        assert "position_ids" in input_names
        assert "logits" in output_names

        # KV cache inputs/outputs
        for i in range(TINY_LAYERS):
            assert f"past_key_values.{i}.key" in input_names
            assert f"past_key_values.{i}.value" in input_names
            assert f"present.{i}.key" in output_names
            assert f"present.{i}.value" in output_names

    def test_whisper_encoder_has_initializers(self):
        """Verify encoder has conv and layer norm initializers."""
        from mobius._builder import build_from_module
        from mobius.models.whisper import WhisperForConditionalGeneration
        from mobius.tasks import SpeechToTextTask

        config = self._whisper_config()
        module = WhisperForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=SpeechToTextTask())
        encoder = pkg["encoder"]

        init_names = list(encoder.graph.initializers)
        assert any("conv1" in n for n in init_names), "Should have conv1 initializers"
        assert any("conv2" in n for n in init_names), "Should have conv2 initializers"
        assert any("self_attn" in n for n in init_names), "Should have attention initializers"
        assert any("layer_norm" in n for n in init_names), "Should have LayerNorm initializer"

    def test_whisper_decoder_has_initializers(self):
        """Verify decoder has embedding, attention, cross-attention, and proj_out initializers."""
        from mobius._builder import build_from_module
        from mobius.models.whisper import WhisperForConditionalGeneration
        from mobius.tasks import SpeechToTextTask

        config = self._whisper_config()
        module = WhisperForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=SpeechToTextTask())
        decoder = pkg["decoder"]

        init_names = list(decoder.graph.initializers)
        assert any("embed_tokens" in n for n in init_names), "Should have token embeddings"
        assert any("embed_positions" in n for n in init_names), (
            "Should have position embeddings"
        )
        assert any("self_attn" in n for n in init_names), "Should have self-attention"
        assert any("encoder_attn" in n for n in init_names), "Should have cross-attention"
        assert any("proj_out" in n for n in init_names), "Should have proj_out"

    def test_whisper_registry_lookup(self):
        """Verify whisper model_type is properly registered."""
        model_cls = registry.get("whisper")
        from mobius.models.whisper import WhisperForConditionalGeneration

        assert model_cls is WhisperForConditionalGeneration


class TestBuildGraphMoonshine:
    """Verify Moonshine raw-audio encoder and cached decoder graphs."""

    def _moonshine_config(self):
        from mobius._configs import MoonshineConfig

        return MoonshineConfig(
            vocab_size=512,
            hidden_size=TINY_HIDDEN,
            intermediate_size=TINY_INTERMEDIATE,
            num_hidden_layers=TINY_LAYERS,
            num_attention_heads=TINY_HEADS,
            num_key_value_heads=TINY_HEADS,
            head_dim=TINY_HIDDEN // TINY_HEADS,
            hidden_act="silu",
            pad_token_id=2,
            tie_word_embeddings=True,
            max_position_embeddings=194,
            rope_type="default",
            rope_theta=10_000.0,
            partial_rotary_factor=0.75,
            rope_interleave=True,
            encoder_num_hidden_layers=TINY_LAYERS,
            encoder_num_attention_heads=TINY_HEADS,
            encoder_num_key_value_heads=TINY_HEADS,
        )

    def test_moonshine_package_and_io(self):
        from mobius._builder import build_from_module
        from mobius.models import MoonshineForConditionalGeneration
        from mobius.tasks import SpeechToTextTask

        config = self._moonshine_config()
        package = build_from_module(
            MoonshineForConditionalGeneration(config),
            config,
            task=SpeechToTextTask(),
        )

        assert set(package) == {"encoder", "decoder"}
        encoder_inputs = {value.name for value in package["encoder"].graph.inputs}
        encoder_outputs = {value.name for value in package["encoder"].graph.outputs}
        decoder_inputs = {value.name for value in package["decoder"].graph.inputs}
        assert encoder_inputs == {"input_values", "attention_mask"}
        assert encoder_outputs == {
            "encoder_hidden_states",
            "encoder_attention_mask",
        }
        assert "encoder_attention_mask" in decoder_inputs
        assert "position_ids" in decoder_inputs
        for layer_idx in range(TINY_LAYERS):
            assert f"past_key_values.{layer_idx}.key" in decoder_inputs

    def test_moonshine_architecture_initializers(self):
        from mobius._builder import build_from_module
        from mobius.models import MoonshineForConditionalGeneration
        from mobius.tasks import SpeechToTextTask

        config = self._moonshine_config()
        package = build_from_module(
            MoonshineForConditionalGeneration(config),
            config,
            task=SpeechToTextTask(),
        )
        encoder_initializers = set(package["encoder"].graph.initializers)
        decoder_initializers = set(package["decoder"].graph.initializers)

        assert "encoder.conv1.weight" in encoder_initializers
        assert "encoder.conv1.bias" not in encoder_initializers
        assert "encoder.groupnorm.weight" in encoder_initializers
        assert "encoder.layers.0.input_layernorm.weight" in encoder_initializers
        assert "encoder.layers.0.input_layernorm.bias" not in encoder_initializers
        assert "encoder.layers.0.self_attn.q_proj.weight" in encoder_initializers
        assert "encoder.layers.0.self_attn.q_proj.bias" not in encoder_initializers
        assert "encoder.layers.0.mlp.fc1.bias" in encoder_initializers
        assert "encoder.layers.0.encoder_attn.q_proj.weight" not in encoder_initializers

        assert "decoder.embed_tokens.weight" in decoder_initializers
        assert "decoder.layers.0.encoder_attn.q_proj.weight" in decoder_initializers
        assert "decoder.layers.0.mlp.fc1.weight" in decoder_initializers
        assert "decoder.proj_out.weight" in decoder_initializers


class TestBuildGraphQwen3ASR:
    """Verify Qwen3-ASR 3-model split with SpeechLanguageTask."""

    def _asr_config(self):
        return _base_config(
            attn_qk_norm=True,
            hidden_act="silu",
            mrope_section=[24, 20, 20],
            mrope_interleaved=True,
            audio=AudioConfig(
                d_model=64,
                encoder_layers=2,
                encoder_attention_heads=4,
                encoder_ffn_dim=128,
                num_mel_bins=128,
                max_source_positions=256,
                downsample_hidden_size=32,
                output_dim=64,
                audio_token_id=100,
            ),
        )

    def test_package_builds_3_models(self):
        """Build Qwen3-ASR and verify 3-model package."""
        from mobius.models.qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )
        from mobius.tasks import SpeechLanguageTask

        config = self._asr_config()
        module = Qwen3ASRForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=SpeechLanguageTask())

        assert "audio_encoder" in pkg
        assert "embedding" in pkg
        assert "decoder" in pkg

    def test_audio_encoder_io(self):
        """Verify audio encoder inputs/outputs."""
        from mobius.models.qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )
        from mobius.tasks import SpeechLanguageTask

        config = self._asr_config()
        module = Qwen3ASRForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=SpeechLanguageTask())
        encoder = pkg["audio_encoder"]

        input_names = {inp.name for inp in encoder.graph.inputs}
        assert "input_features" in input_names
        # feature_attention_mask is required so the encoder can ignore
        # padded mel frames; without it the LLM emits degenerate loops
        # on any input padded by the standard HF processor.
        assert "feature_attention_mask" in input_names

        output_names = {out.name for out in encoder.graph.outputs}
        assert "audio_features" in output_names
        # audio_feature_lengths exposes the valid token count after
        # the encoder's 8x time downsampling so downstream callers can
        # crop padding-derived rows out of audio_features before the
        # embedding gather.
        assert "audio_feature_lengths" in output_names

    def test_audio_encoder_attention_uses_mask(self):
        """The encoder's Attention ops must receive the mask input.

        Guards against accidentally dropping the mask wiring inside
        the encoder forward — the graph builds without it but the
        encoder behaves the same as the pre-fix version.
        """
        from mobius.models.qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )
        from mobius.tasks import SpeechLanguageTask

        config = self._asr_config()
        module = Qwen3ASRForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=SpeechLanguageTask())
        encoder = pkg["audio_encoder"]

        attention_nodes = [n for n in encoder.graph if n.op_type == "Attention"]
        assert attention_nodes, "audio encoder must contain Attention ops"
        for node in attention_nodes:
            # 4th positional input on op.Attention is attn_mask; must
            # be a wired value, not None / empty.
            assert len(node.inputs) >= 4
            assert node.inputs[3] is not None

    def test_embedding_io(self):
        """Verify embedding model inputs/outputs."""
        from mobius.models.qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )
        from mobius.tasks import SpeechLanguageTask

        config = self._asr_config()
        module = Qwen3ASRForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=SpeechLanguageTask())
        embedding = pkg["embedding"]

        input_names = {inp.name for inp in embedding.graph.inputs}
        assert "input_ids" in input_names
        assert "audio_features" in input_names

        output_names = {out.name for out in embedding.graph.outputs}
        assert "inputs_embeds" in output_names

    def test_decoder_io(self):
        """Verify decoder has MRoPE position_ids and KV cache."""
        from mobius.models.qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )
        from mobius.tasks import SpeechLanguageTask

        config = self._asr_config()
        module = Qwen3ASRForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=SpeechLanguageTask())
        decoder = pkg["decoder"]

        input_names = {inp.name for inp in decoder.graph.inputs}
        assert "inputs_embeds" in input_names
        assert "attention_mask" in input_names
        assert "position_ids" in input_names

        output_names = {out.name for out in decoder.graph.outputs}
        assert "logits" in output_names
        assert "present.0.key" in output_names
        assert "present.0.value" in output_names

    def test_registry_lookup(self):
        """Verify qwen3_asr is registered with speech-language task."""
        model_cls = registry.get("qwen3_asr")
        from mobius.models.qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )

        assert model_cls is Qwen3ASRForConditionalGeneration
        assert _default_task_for_model("qwen3_asr") == "speech-language"

    def test_qwen3_forced_aligner_alias_resolves(self):
        """Verify qwen3_forced_aligner alias resolves to same class as qwen3_asr."""
        from mobius.models.qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )

        assert registry.get("qwen3_forced_aligner") is Qwen3ASRForConditionalGeneration
        assert registry.get("qwen3_forced_aligner") is registry.get("qwen3_asr")
        assert _default_task_for_model("qwen3_forced_aligner") == "speech-language"

    def test_3model_pipeline_runs_with_ort(self):
        """Run audio_encoder → embedding with ORT.

        Guards against audio token count mismatches: the number of
        AUDIO_TOKEN_ID positions in input_ids must equal the number of
        audio feature rows from the encoder, otherwise the embedding
        Gather goes out of bounds.
        """
        import numpy as np

        from mobius._testing.ort_inference import (
            OnnxModelSession,
        )
        from mobius.models.qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )
        from mobius.rewrite_rules._testing_utils import (
            fill_random_weights,
        )
        from mobius.tasks import SpeechLanguageTask

        config = self._asr_config()
        module = Qwen3ASRForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=SpeechLanguageTask())

        for model in pkg.values():
            fill_random_weights(model)

        # Step 1: Audio encoder — random mel input
        enc_sess = OnnxModelSession(pkg["audio_encoder"])
        mel_seq = 100
        mel = np.random.randn(1, config.audio.num_mel_bins, mel_seq).astype(np.float32)
        # Mark the last 20 mel frames as padding to exercise the mask
        # path. The encoder must crop the corresponding audio rows so
        # they don't leak into the embedding's Gather.
        feature_attention_mask = np.ones((1, mel_seq), dtype=np.int64)
        feature_attention_mask[:, -20:] = 0
        enc_out = enc_sess.run(
            {
                "input_features": mel,
                "feature_attention_mask": feature_attention_mask,
            }
        )
        audio_features = enc_out["audio_features"]
        audio_feature_lengths = enc_out["audio_feature_lengths"]
        # Crop padding-derived rows before passing to the embedding —
        # this mirrors what production callers must do.
        valid_len = int(audio_feature_lengths[0])
        assert 0 < valid_len <= audio_features.shape[1]
        audio_features = audio_features[:, :valid_len, :]
        num_audio_tokens = audio_features.shape[1]
        # Flatten to 2D: (num_audio_tokens, output_dim)
        audio_features_2d = audio_features.reshape(-1, audio_features.shape[-1])
        enc_sess.close()

        # Step 2: Embedding — mix text + audio tokens
        # Build input_ids with exactly num_audio_tokens audio pad tokens
        # Use the config's audio_token_id (must be within vocab_size)
        audio_token_id = config.audio.audio_token_id
        prefix = [1, 2, 3]  # mock system/user tokens
        suffix = [4, 5]  # mock footer tokens
        input_ids = np.array(
            [prefix + [audio_token_id] * num_audio_tokens + suffix],
            dtype=np.int64,
        )

        embed_sess = OnnxModelSession(pkg["embedding"])
        embed_out = embed_sess.run(
            {
                "input_ids": input_ids,
                "audio_features": audio_features_2d,
            }
        )
        inputs_embeds = embed_out["inputs_embeds"]
        embed_sess.close()

        seq_len = inputs_embeds.shape[1]
        assert seq_len == input_ids.shape[1]
        assert inputs_embeds.shape[2] == config.hidden_size

        # Step 3: Decoder — single forward pass with MRoPE
        decoder_sess = OnnxModelSession(pkg["decoder"])
        past_kv = {}
        for i in range(config.num_hidden_layers):
            past_kv[f"past_key_values.{i}.key"] = np.zeros(
                (1, config.num_key_value_heads, 0, config.head_dim),
                dtype=np.float32,
            )
            past_kv[f"past_key_values.{i}.value"] = np.zeros(
                (1, config.num_key_value_heads, 0, config.head_dim),
                dtype=np.float32,
            )

        pos = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
        # MRoPE: (3, 1, seq_len)
        position_ids = np.stack([pos, pos, pos])

        dec_out = decoder_sess.run(
            {
                "inputs_embeds": inputs_embeds,
                "attention_mask": np.ones((1, seq_len), dtype=np.int64),
                "position_ids": position_ids,
                **past_kv,
            }
        )
        decoder_sess.close()

        logits = dec_out["logits"]
        assert logits.shape[0] == 1
        assert logits.shape[1] == seq_len


class TestBuildGraphFunASR:
    """Verify Fun-ASR-Nano 3-model split with FunASRSpeechLanguageTask."""

    def _fun_asr_config(self):
        return _base_config(
            attn_qk_norm=True,
            hidden_act="silu",
            audio=AudioConfig(
                input_size=32,
                attention_dim=TINY_HIDDEN,
                attention_heads=TINY_HEADS,
                num_blocks=3,
                linear_units=TINY_INTERMEDIATE,
                kernel_size=5,
                tp_num_blocks=2,
                output_dim=TINY_HIDDEN,
                audio_token_id=100,
                adaptor_proj_dim=TINY_INTERMEDIATE,
                adaptor_num_blocks=2,
                adaptor_ffn_dim=32,
                adaptor_num_heads=TINY_HEADS,
            ),
        )

    def test_package_builds_3_models(self):
        """Build Fun-ASR and verify 3-model package."""
        from mobius.models.fun_asr import FunASRForConditionalGeneration
        from mobius.tasks import FunASRSpeechLanguageTask

        config = self._fun_asr_config()
        module = FunASRForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=FunASRSpeechLanguageTask())

        assert "audio_encoder" in pkg
        assert "embedding" in pkg
        assert "decoder" in pkg

    def test_audio_encoder_io(self):
        """Verify audio encoder inputs/outputs."""
        from mobius.models.fun_asr import FunASRForConditionalGeneration
        from mobius.tasks import FunASRSpeechLanguageTask

        config = self._fun_asr_config()
        module = FunASRForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=FunASRSpeechLanguageTask())
        encoder = pkg["audio_encoder"]

        input_names = {inp.name for inp in encoder.graph.inputs}
        assert "input_features" in input_names

        output_names = {out.name for out in encoder.graph.outputs}
        assert "audio_features" in output_names

    def test_embedding_io(self):
        """Verify embedding model inputs/outputs."""
        from mobius.models.fun_asr import FunASRForConditionalGeneration
        from mobius.tasks import FunASRSpeechLanguageTask

        config = self._fun_asr_config()
        module = FunASRForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=FunASRSpeechLanguageTask())
        embedding = pkg["embedding"]

        input_names = {inp.name for inp in embedding.graph.inputs}
        assert "input_ids" in input_names
        assert "audio_features" in input_names

        output_names = {out.name for out in embedding.graph.outputs}
        assert "inputs_embeds" in output_names

    def test_decoder_io(self):
        """Verify decoder has standard position_ids and KV cache."""
        from mobius.models.fun_asr import FunASRForConditionalGeneration
        from mobius.tasks import FunASRSpeechLanguageTask

        config = self._fun_asr_config()
        module = FunASRForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=FunASRSpeechLanguageTask())
        decoder = pkg["decoder"]

        input_names = {inp.name for inp in decoder.graph.inputs}
        assert "inputs_embeds" in input_names
        assert "attention_mask" in input_names
        assert "position_ids" in input_names

        output_names = {out.name for out in decoder.graph.outputs}
        assert "logits" in output_names
        assert "present.0.key" in output_names
        assert "present.0.value" in output_names

    def test_registry_lookup(self):
        """Verify fun_asr is registered with fun-asr-speech-language task."""
        model_cls = registry.get("fun_asr")
        from mobius.models.fun_asr import FunASRForConditionalGeneration

        assert model_cls is FunASRForConditionalGeneration
        assert _default_task_for_model("fun_asr") == "fun-asr-speech-language"

    def test_3model_pipeline_runs_with_ort(self):
        """Run audio_encoder → embedding → decoder with ORT."""
        import numpy as np

        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.models.fun_asr import FunASRForConditionalGeneration
        from mobius.rewrite_rules._testing_utils import fill_random_weights
        from mobius.tasks import FunASRSpeechLanguageTask

        config = self._fun_asr_config()
        module = FunASRForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=FunASRSpeechLanguageTask())

        for model in pkg.values():
            fill_random_weights(model)

        # Step 1: Audio encoder — random fbank input
        # Sequence length must be even (temporal pooling halves it)
        input_dim = config.audio.input_size
        enc_sess = OnnxModelSession(pkg["audio_encoder"])
        fbank = np.random.randn(1, 100, input_dim).astype(np.float32)
        enc_out = enc_sess.run({"input_features": fbank})
        audio_features = enc_out["audio_features"]
        num_audio_tokens = audio_features.shape[1]
        audio_features_2d = audio_features.reshape(-1, audio_features.shape[-1])
        enc_sess.close()

        # Step 2: Embedding — mix text + audio tokens
        audio_token_id = config.audio.audio_token_id
        prefix = [1, 2, 3]
        suffix = [4, 5]
        input_ids = np.array(
            [prefix + [audio_token_id] * num_audio_tokens + suffix],
            dtype=np.int64,
        )

        embed_sess = OnnxModelSession(pkg["embedding"])
        embed_out = embed_sess.run(
            {
                "input_ids": input_ids,
                "audio_features": audio_features_2d,
            }
        )
        inputs_embeds = embed_out["inputs_embeds"]
        embed_sess.close()

        seq_len = inputs_embeds.shape[1]
        assert seq_len == input_ids.shape[1]
        assert inputs_embeds.shape[2] == config.hidden_size

        # Step 3: Decoder — single forward pass
        decoder_sess = OnnxModelSession(pkg["decoder"])
        past_kv = {}
        for i in range(config.num_hidden_layers):
            past_kv[f"past_key_values.{i}.key"] = np.zeros(
                (1, config.num_key_value_heads, 0, config.head_dim),
                dtype=np.float32,
            )
            past_kv[f"past_key_values.{i}.value"] = np.zeros(
                (1, config.num_key_value_heads, 0, config.head_dim),
                dtype=np.float32,
            )

        pos = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
        dec_out = decoder_sess.run(
            {
                "inputs_embeds": inputs_embeds,
                "attention_mask": np.ones((1, seq_len), dtype=np.int64),
                "position_ids": pos,
                **past_kv,
            }
        )
        decoder_sess.close()

        logits = dec_out["logits"]
        assert logits.shape[0] == 1
        assert logits.shape[1] == seq_len


class TestBuildGraphQwen3TTS:
    """Verify Qwen3-TTS 4-model split with TTSTask."""

    def _tts_config(self):
        """Tiny config mimicking 0.6B: hidden_size=64 but text_hidden_size=128."""
        return _base_config(
            attn_qk_norm=True,
            hidden_act="silu",
            rope_scaling={
                "rope_type": "default",
                "mrope_section": [24, 20, 20],
            },
            mrope_interleaved=True,
            tts=TTSConfig(
                text_hidden_size=TINY_INTERMEDIATE,  # 128 (larger than hidden)
                text_vocab_size=TINY_VOCAB,
                num_code_groups=4,  # Fewer groups for testing
                code_predictor=CodePredictorConfig(
                    hidden_size=TINY_HIDDEN,
                    intermediate_size=TINY_INTERMEDIATE,
                    num_hidden_layers=2,
                    num_attention_heads=TINY_HEADS,
                    num_key_value_heads=TINY_KV_HEADS,
                    head_dim=TINY_HEAD_DIM,
                    vocab_size=TINY_VOCAB,
                    num_code_groups=4,
                ),
                speaker_encoder=SpeakerEncoderConfig(
                    mel_dim=32,
                    enc_dim=TINY_HIDDEN,
                    enc_channels=[16, 16, 16, 16, 48],
                    enc_kernel_sizes=[5, 3, 3, 3, 1],
                    enc_dilations=[1, 2, 3, 4, 1],
                    enc_attention_channels=16,
                    enc_res2net_scale=2,
                    enc_se_channels=16,
                ),
            ),
        )

    def test_package_builds_4_models(self):
        """Build Qwen3-TTS and verify 4-model package."""
        from mobius.models.qwen3_tts import (
            Qwen3TTSForConditionalGeneration,
        )
        from mobius.tasks import TTSTask

        config = self._tts_config()
        module = Qwen3TTSForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=TTSTask())

        assert "talker" in pkg
        assert "code_predictor" in pkg
        assert "embedding" in pkg
        assert "speaker_encoder" in pkg

    def test_talker_io(self):
        """Verify talker has inputs_embeds, logits, last_hidden_state, KV cache."""
        from mobius.models.qwen3_tts import (
            Qwen3TTSForConditionalGeneration,
        )
        from mobius.tasks import TTSTask

        config = self._tts_config()
        module = Qwen3TTSForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=TTSTask())
        talker = pkg["talker"]

        input_names = {inp.name for inp in talker.graph.inputs}
        assert "inputs_embeds" in input_names
        assert "attention_mask" in input_names
        assert "position_ids" in input_names
        assert "past_key_values.0.key" in input_names

        output_names = {out.name for out in talker.graph.outputs}
        assert "logits" in output_names
        assert "last_hidden_state" in output_names
        assert "present.0.key" in output_names

    def test_code_predictor_io(self):
        """Verify code predictor takes inputs_embeds and step_index."""
        from mobius.models.qwen3_tts import (
            Qwen3TTSForConditionalGeneration,
        )
        from mobius.tasks import TTSTask

        config = self._tts_config()
        module = Qwen3TTSForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=TTSTask())
        cp = pkg["code_predictor"]

        input_names = {inp.name for inp in cp.graph.inputs}
        assert "inputs_embeds" in input_names
        assert "step_index" in input_names
        assert "position_ids" in input_names
        assert "attention_mask" in input_names

        output_names = {out.name for out in cp.graph.outputs}
        assert "logits" in output_names

        # Verify 2D position_ids (1D RoPE, not 3D MRoPE)
        pos_input = next(i for i in cp.graph.inputs if i.name == "position_ids")
        assert len(pos_input.shape) == 2  # (batch, seq_len)

    def test_embedding_io(self):
        """Verify embedding model takes text_ids + codec_ids."""
        from mobius.models.qwen3_tts import (
            Qwen3TTSForConditionalGeneration,
        )
        from mobius.tasks import TTSTask

        config = self._tts_config()
        module = Qwen3TTSForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=TTSTask())
        embedding = pkg["embedding"]

        input_names = {inp.name for inp in embedding.graph.inputs}
        assert "text_ids" in input_names
        assert "codec_ids" in input_names

        output_names = {out.name for out in embedding.graph.outputs}
        assert "text_embeds" in output_names
        assert "codec_embeds" in output_names

    def test_speaker_encoder_io(self):
        """Verify speaker encoder takes mel_input."""
        from mobius.models.qwen3_tts import (
            Qwen3TTSForConditionalGeneration,
        )
        from mobius.tasks import TTSTask

        config = self._tts_config()
        module = Qwen3TTSForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=TTSTask())
        se = pkg["speaker_encoder"]

        input_names = {inp.name for inp in se.graph.inputs}
        assert "mel_input" in input_names

        output_names = {out.name for out in se.graph.outputs}
        assert "speaker_embedding" in output_names

    def test_registry_lookup(self):
        """Verify qwen3_tts is registered with tts task."""
        model_cls = registry.get("qwen3_tts")
        from mobius.models.qwen3_tts import (
            Qwen3TTSForConditionalGeneration,
        )

        assert model_cls is Qwen3TTSForConditionalGeneration
        assert _default_task_for_model("qwen3_tts") == "tts"


class TestBuildVAEGraph:
    """Verify VAE (AutoencoderKL) graph construction."""

    def _vae_config(self):
        from mobius._diffusers_configs import VAEConfig

        return VAEConfig(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            block_out_channels=(32, 64),
            layers_per_block=1,
            norm_num_groups=32,
            act_fn="silu",
            mid_block_add_attention=True,
            use_quant_conv=True,
            use_post_quant_conv=True,
        )

    def test_decoder_graph_builds(self):
        from mobius.models.vae import AutoencoderKLModel
        from mobius.tasks import VAETask

        config = self._vae_config()
        module = AutoencoderKLModel(config)
        task = VAETask()
        pkg = task.build(module, config)
        model = pkg["decoder"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "latent_sample" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "sample" in output_names

    def test_package_has_encoder_and_decoder(self):
        from mobius.models.vae import AutoencoderKLModel
        from mobius.tasks import VAETask

        config = self._vae_config()
        module = AutoencoderKLModel(config)
        task = VAETask()
        pkg = task.build(module, config)

        assert "encoder" in pkg
        assert "decoder" in pkg

        # Encoder: sample → latent_dist
        enc_inputs = {inp.name for inp in pkg["encoder"].graph.inputs}
        enc_outputs = {out.name for out in pkg["encoder"].graph.outputs}
        assert "sample" in enc_inputs
        assert "latent_dist" in enc_outputs

        # Decoder: latent_sample → sample
        dec_inputs = {inp.name for inp in pkg["decoder"].graph.inputs}
        dec_outputs = {out.name for out in pkg["decoder"].graph.outputs}
        assert "latent_sample" in dec_inputs
        assert "sample" in dec_outputs

    def test_decoder_has_initializers(self):
        from mobius.models.vae import AutoencoderKLModel
        from mobius.tasks import VAETask

        config = self._vae_config()
        module = AutoencoderKLModel(config)
        task = VAETask()
        pkg = task.build(module, config)
        model = pkg["decoder"]

        init_names = list(model.graph.initializers)
        assert len(init_names) > 0
        has_conv = any("conv" in n for n in init_names)
        has_norm = any("norm" in n for n in init_names)
        assert has_conv, "Should have conv initializers"
        assert has_norm, "Should have norm initializers"


class TestBuildAudioGraph:
    """Verify audio encoder-only models build valid ONNX graphs."""

    def test_wav2vec2_graph_builds(self):
        from mobius.models.wav2vec2 import Wav2Vec2Model
        from mobius.tasks import AudioFeatureExtractionTask

        config = _base_config()
        module = Wav2Vec2Model(config)
        task = AudioFeatureExtractionTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "input_values" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "last_hidden_state" in output_names

    def test_wav2vec2_has_initializers(self):
        from mobius.models.wav2vec2 import Wav2Vec2Model
        from mobius.tasks import AudioFeatureExtractionTask

        config = _base_config()
        module = Wav2Vec2Model(config)
        task = AudioFeatureExtractionTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        init_names = list(model.graph.initializers)
        has_feature_extractor = any("feature_extractor" in n for n in init_names)
        has_attention = any("attention" in n for n in init_names)
        assert has_feature_extractor, "Should have feature extractor initializers"
        assert has_attention, "Should have attention initializers"

    def test_audio_aliases_build(self):
        """Audio model aliases (hubert, wavlm, musicgen, etc.) all build."""
        from mobius.tasks import AudioFeatureExtractionTask

        config = _base_config()
        task = AudioFeatureExtractionTask()
        for model_type in (
            "data2vec-audio",
            "hubert",
            "wavlm",
            "mctct",
            "musicgen",
            "seamless_m4t",
            "seamless_m4t_v2",
            "sew",
            "sew-d",
            "speecht5",
            "unispeech",
            "unispeech-sat",
            "voxtral_encoder",
            "wav2vec2",
            "wav2vec2-bert",
            "wav2vec2-conformer",
        ):
            model_cls = registry.get(model_type)
            module = model_cls(config)
            pkg = task.build(module, config)
            model = pkg["model"]
            assert model.graph is not None, f"{model_type} graph should build"

            input_names = {inp.name for inp in model.graph.inputs}
            assert "input_values" in input_names, f"{model_type} missing input_values"

            output_names = {out.name for out in model.graph.outputs}
            assert "last_hidden_state" in output_names, (
                f"{model_type} missing last_hidden_state"
            )

            init_names = list(model.graph.initializers)
            assert len(init_names) > 0, f"{model_type} should have initializers"


class TestBuildMMSGraph:
    """Verify MMS (Massively Multilingual Speech) CTC model builds correctly.

    Tests both the base wav2vec2 encoder + CTC head, and with the per-language
    adapter (``add_adapter=True``) that enables language switching in MMS-1b-all.
    """

    def _mms_config(self, add_adapter: bool = False):
        """Tiny CTC config: hidden=64, 2 layers, 10 vocab labels."""
        return _base_config(
            config_cls=MMSConfig,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            vocab_size=10,
            add_adapter=add_adapter,
            output_hidden_size=64,
            adapter_kernel_size=3,
            adapter_stride=2,
            num_adapter_layers=2,
        )

    def test_package_builds(self):
        """Build MMS ONNX model and verify single-model package."""
        from mobius.models.wav2vec2_ctc import Wav2Vec2ForCTCModel
        from mobius.tasks import CTCAsrTask

        config = self._mms_config()
        module = Wav2Vec2ForCTCModel(config)
        pkg = CTCAsrTask().build(module, config)

        assert "model" in pkg

    def test_io_contract(self):
        """Verify input/output names of the CTC model."""
        from mobius.models.wav2vec2_ctc import Wav2Vec2ForCTCModel
        from mobius.tasks import CTCAsrTask

        config = self._mms_config()
        module = Wav2Vec2ForCTCModel(config)
        pkg = CTCAsrTask().build(module, config)
        model = pkg["model"]

        input_names = {inp.name for inp in model.graph.inputs}
        assert "input_values" in input_names
        assert "attention_mask" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "logits" in output_names

    def test_has_ctc_head_initializers(self):
        """Verify the CTC lm_head parameters are present."""
        from mobius.models.wav2vec2_ctc import Wav2Vec2ForCTCModel
        from mobius.tasks import CTCAsrTask

        config = self._mms_config()
        module = Wav2Vec2ForCTCModel(config)
        pkg = CTCAsrTask().build(module, config)
        model = pkg["model"]

        init_names = list(model.graph.initializers)
        assert any("lm_head" in n for n in init_names), "Should have lm_head params"
        assert any("feature_extractor" in n for n in init_names), (
            "Should have feature_extractor params"
        )
        assert any("encoder" in n for n in init_names), "Should have encoder params"

    def test_adapter_variant_builds(self):
        """Build with adapter enabled (MMS-1b-all language adapter path)."""
        from mobius.models.wav2vec2_ctc import Wav2Vec2ForCTCModel
        from mobius.tasks import CTCAsrTask

        config = self._mms_config(add_adapter=True)
        module = Wav2Vec2ForCTCModel(config)
        pkg = CTCAsrTask().build(module, config)
        model = pkg["model"]

        init_names = list(model.graph.initializers)
        assert any("adapter" in n for n in init_names), (
            "Should have adapter params when add_adapter=True"
        )

    def test_registry_lookup(self):
        """Verify 'mms' is registered with ctc-asr task."""
        from mobius.models.wav2vec2_ctc import Wav2Vec2ForCTCModel

        assert registry.get("mms") is Wav2Vec2ForCTCModel
        assert _default_task_for_model("mms") == "ctc-asr"

    def test_ort_inference(self):
        """Build and run MMS through OnnxRuntime end-to-end."""
        import numpy as np

        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.models.wav2vec2_ctc import Wav2Vec2ForCTCModel
        from mobius.rewrite_rules._testing_utils import fill_random_weights
        from mobius.tasks import CTCAsrTask

        config = self._mms_config()
        module = Wav2Vec2ForCTCModel(config)
        pkg = CTCAsrTask().build(module, config)
        fill_random_weights(pkg["model"])

        sess = OnnxModelSession(pkg["model"])
        num_samples = 8000  # 0.5 sec at 16 kHz
        waveform = np.random.randn(1, num_samples).astype(np.float32)
        attention_mask = np.ones((1, num_samples), dtype=np.int64)

        out = sess.run({"input_values": waveform, "attention_mask": attention_mask})
        sess.close()

        logits = out["logits"]
        assert logits.shape[0] == 1  # batch
        assert logits.shape[1] > 0  # num_frames (after CNN downsampling)
        assert logits.shape[2] == config.vocab_size  # CTC vocab

    """Verify UNet2DConditionModel graph construction."""

    def _unet_config(self):
        from mobius._diffusers_configs import UNet2DConfig

        return UNet2DConfig(
            in_channels=4,
            out_channels=4,
            block_out_channels=(32, 64),
            layers_per_block=1,
            norm_num_groups=32,
            cross_attention_dim=32,
            attention_head_dim=8,
        )

    def test_unet_graph_builds(self):
        from mobius.models.unet import UNet2DConditionModel
        from mobius.tasks import DenoisingTask

        config = self._unet_config()
        module = UNet2DConditionModel(config)
        task = DenoisingTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "sample" in input_names
        assert "timestep" in input_names
        assert "encoder_hidden_states" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "noise_pred" in output_names

    def test_unet_has_initializers(self):
        from mobius.models.unet import UNet2DConditionModel
        from mobius.tasks import DenoisingTask

        config = self._unet_config()
        module = UNet2DConditionModel(config)
        task = DenoisingTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        init_names = list(model.graph.initializers)
        assert len(init_names) > 0
        has_time_emb = any("time_embedding" in n for n in init_names)
        has_conv = any("conv" in n for n in init_names)
        has_mid = any("mid_block" in n for n in init_names)
        assert has_time_emb, "Should have time embedding initializers"
        assert has_conv, "Should have conv initializers"
        assert has_mid, "Should have mid block initializers"


class TestBuildDiTGraph:
    """Verify DiT transformer denoiser graph construction."""

    def test_dit_graph_builds(self):
        from mobius.models.dit import DiTConfig, DiTTransformer2DModel
        from mobius.tasks import DenoisingTask

        config = DiTConfig(
            in_channels=4,
            out_channels=4,
            patch_size=2,
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            cross_attention_dim=32,
            caption_channels=32,
            sample_size=8,
        )
        module = DiTTransformer2DModel(config)
        task = DenoisingTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "sample" in input_names
        assert "timestep" in input_names
        assert "encoder_hidden_states" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "noise_pred" in output_names


class TestBuildHunyuanDiTGraph:
    """Verify HunyuanDiT transformer denoiser graph construction."""

    def test_hunyuan_dit_graph_builds(self):
        from mobius.models.hunyuan_dit import (
            HunyuanDiT2DModel,
            HunyuanDiTConfig,
        )
        from mobius.tasks import DenoisingTask

        config = HunyuanDiTConfig(
            in_channels=4,
            patch_size=2,
            hidden_size=64,
            num_layers=4,
            num_attention_heads=4,
            cross_attention_dim=32,
            mlp_ratio=4.0,
            learn_sigma=True,
            sample_size=8,
        )
        module = HunyuanDiT2DModel(config)
        task = DenoisingTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "sample" in input_names
        assert "timestep" in input_names
        assert "encoder_hidden_states" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "noise_pred" in output_names


class TestBuildControlNetGraph:
    """Verify ControlNet model graph construction."""

    def test_controlnet_graph_builds(self):
        from mobius.models.controlnet import ControlNetConfig, ControlNetModel
        from mobius.tasks import ControlNetTask

        config = ControlNetConfig(
            in_channels=4,
            conditioning_channels=3,
            block_out_channels=(32, 64),
            layers_per_block=1,
            norm_num_groups=32,
            cross_attention_dim=32,
            attention_head_dim=8,
        )
        module = ControlNetModel(config)
        task = ControlNetTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "sample" in input_names
        assert "timestep" in input_names
        assert "encoder_hidden_states" in input_names
        assert "controlnet_cond" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "mid_block_res" in output_names
        down_res = [n for n in output_names if n.startswith("down_block_res_")]
        assert len(down_res) > 0, "Should have down block residual outputs"


class TestBuildVideoVAEGraph:
    """Verify Video VAE (3D autoencoder) graph construction."""

    def test_video_decoder_graph_builds(self):
        from mobius.models.video_vae import VideoAutoencoderModel, VideoVAEConfig
        from mobius.tasks import VAETask

        config = VideoVAEConfig(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            block_out_channels=(32, 64),
            layers_per_block=1,
            norm_num_groups=32,
        )
        module = VideoAutoencoderModel(config)
        task = VAETask()
        pkg = task.build(module, config)
        model = pkg["decoder"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "latent_sample" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "sample" in output_names

        init_names = list(model.graph.initializers)
        assert len(init_names) > 0
        has_conv = any("conv" in n for n in init_names)
        assert has_conv, "Should have 3D conv initializers"


class TestBuildSD3Graph:
    """Verify SD3 (MMDiT) transformer denoiser graph construction."""

    def test_sd3_graph_builds(self):
        from mobius.models.flux_sd3 import SD3Config, SD3Transformer2DModel
        from mobius.tasks import DenoisingTask

        config = SD3Config(
            in_channels=4,
            out_channels=4,
            patch_size=2,
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            joint_attention_dim=32,
            caption_projection_dim=32,
            cross_attention_dim=32,
            sample_size=8,
        )
        module = SD3Transformer2DModel(config)
        task = DenoisingTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "sample" in input_names
        assert "timestep" in input_names
        assert "encoder_hidden_states" in input_names
        assert "noise_pred" in {out.name for out in model.graph.outputs}


class TestBuildFluxGraph:
    """Verify Flux transformer denoiser graph construction."""

    def test_flux_graph_builds(self):
        from mobius.models.flux_sd3 import FluxConfig, FluxTransformer2DModel
        from mobius.tasks import DenoisingTask

        config = FluxConfig(
            in_channels=4,
            out_channels=4,
            patch_size=2,
            hidden_size=64,
            num_layers=1,
            num_single_layers=2,
            num_attention_heads=4,
            joint_attention_dim=32,
            cross_attention_dim=32,
            sample_size=8,
        )
        module = FluxTransformer2DModel(config)
        task = DenoisingTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "sample" in input_names
        assert "timestep" in input_names
        assert "encoder_hidden_states" in input_names
        assert "noise_pred" in {out.name for out in model.graph.outputs}


class TestBuildCogVideoXGraph:
    """Verify CogVideoX 3D video transformer graph construction."""

    def test_cogvideox_graph_builds(self):
        from mobius._diffusers_configs import CogVideoXConfig
        from mobius.models.cogvideox import CogVideoXTransformer3DModel
        from mobius.tasks import VideoDenoisingTask

        config = CogVideoXConfig(
            num_attention_heads=2,
            attention_head_dim=32,
            in_channels=4,
            out_channels=4,
            time_embed_dim=64,
            text_embed_dim=32,
            num_layers=2,
            patch_size=2,
            sample_height=8,
            sample_width=8,
            sample_frames=9,
            temporal_compression_ratio=4,
            max_text_seq_length=8,
            spatial_interpolation_scale=1.0,
            temporal_interpolation_scale=1.0,
            norm_eps=1e-5,
            cross_attention_dim=32,
        )
        module = CogVideoXTransformer3DModel(config)
        task = VideoDenoisingTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "sample" in input_names
        assert "timestep" in input_names
        assert "encoder_hidden_states" in input_names
        assert "noise_pred" in {out.name for out in model.graph.outputs}

        # Verify 5D sample shape
        sample_input = next(inp for inp in model.graph.inputs if inp.name == "sample")
        assert len(sample_input.shape) == 5  # [B, T, C, H, W]


class TestBuildAdapterGraph:
    """Verify T2I-Adapter and IP-Adapter graph construction."""

    def test_t2i_adapter_graph_builds(self):
        from mobius.models.adapters import T2IAdapterConfig, T2IAdapterModel
        from mobius.tasks import AdapterTask

        config = T2IAdapterConfig(in_channels=3, channels=(32, 64), num_res_blocks=1)
        module = T2IAdapterModel(config)
        task = AdapterTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "condition" in input_names
        output_names = {out.name for out in model.graph.outputs}
        assert any(n.startswith("feature_") for n in output_names)

    def test_ip_adapter_graph_builds(self):
        from mobius.models.adapters import IPAdapterConfig, IPAdapterModel
        from mobius.tasks import AdapterTask

        config = IPAdapterConfig(image_embed_dim=32, cross_attention_dim=64, num_tokens=4)
        module = IPAdapterModel(config)
        task = AdapterTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "image_embeds" in input_names
        output_names = {out.name for out in model.graph.outputs}
        assert "adapter_output" in output_names


class TestBuildQwenImageGraph:
    """Verify QwenImage transformer denoiser graph construction."""

    def test_qwen_image_transformer_graph_builds(self):
        from mobius._diffusers_configs import QwenImageConfig
        from mobius.models.qwen_image import QwenImageTransformer2DModel
        from mobius.tasks import DenoisingTask

        config = QwenImageConfig(
            in_channels=4,
            out_channels=4,
            patch_size=2,
            num_layers=2,
            attention_head_dim=32,
            num_attention_heads=2,
            joint_attention_dim=64,
            cross_attention_dim=64,
        )
        module = QwenImageTransformer2DModel(config)
        task = DenoisingTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "sample" in input_names
        assert "timestep" in input_names
        assert "encoder_hidden_states" in input_names
        assert "noise_pred" in {out.name for out in model.graph.outputs}

    def test_qwen_image_vae_encoder_decoder_graphs_build(self):
        from mobius._diffusers_configs import QwenImageVAEConfig
        from mobius.models.qwen_image_vae import AutoencoderKLQwenImageModel
        from mobius.tasks import QwenImageVAETask

        config = QwenImageVAEConfig(
            base_dim=8,
            z_dim=4,
            dim_mult=(1, 2),
            num_res_blocks=1,
            temperal_downsample=(False,),
        )
        module = AutoencoderKLQwenImageModel(config)
        task = QwenImageVAETask()
        pkg = task.build(module, config)

        enc = pkg["encoder"]
        assert enc.graph is not None
        assert "sample" in {inp.name for inp in enc.graph.inputs}
        assert "latent_dist" in {out.name for out in enc.graph.outputs}

        dec = pkg["decoder"]
        assert dec.graph is not None
        assert "latent_sample" in {inp.name for inp in dec.graph.inputs}
        assert "sample" in {out.name for out in dec.graph.outputs}


class TestBuildMimiCodec:
    """Verify the Mimi codec (nvidia/personaplex-7b-v1) graph construction."""

    def test_package_builds_2_models(self):
        """Build the Mimi codec and verify a 2-model (encoder+decoder) package."""
        from mobius.models.mimi import MimiModel, mimi_default_config
        from mobius.tasks import CodecTask

        config = mimi_default_config()
        module = MimiModel(config)
        pkg = build_from_module(module, config, task=CodecTask())

        assert "encoder" in pkg
        assert "decoder" in pkg

    def test_encoder_io(self):
        """Verify the Mimi encoder I/O contract: waveform -> codes."""
        from mobius.models.mimi import MimiModel, mimi_default_config
        from mobius.tasks import CodecTask

        config = mimi_default_config()
        pkg = build_from_module(MimiModel(config), config, task=CodecTask())
        encoder = pkg["encoder"]

        assert "waveform" in {inp.name for inp in encoder.graph.inputs}
        assert "codes" in {out.name for out in encoder.graph.outputs}

    def test_decoder_io(self):
        """Verify the Mimi decoder I/O contract: codes -> waveform."""
        from mobius.models.mimi import MimiModel, mimi_default_config
        from mobius.tasks import CodecTask

        config = mimi_default_config()
        pkg = build_from_module(MimiModel(config), config, task=CodecTask())
        decoder = pkg["decoder"]

        assert "codes" in {inp.name for inp in decoder.graph.inputs}
        assert "waveform" in {out.name for out in decoder.graph.outputs}


class TestBuildMoshiLM:
    """Verify the Moshi LM (nvidia/personaplex-7b-v1) graph construction."""

    @staticmethod
    def _temporal_tiny_config():
        import dataclasses

        from mobius.models.moshi import moshi_temporal_config

        cfg = moshi_temporal_config()
        return dataclasses.replace(
            cfg,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            intermediate_size=128,
            max_position_embeddings=256,
        )

    def test_temporal_io(self):
        """Temporal model I/O: 17-channel frame -> hidden + text_logits + KV."""
        from mobius.models.moshi import MoshiTemporalModel
        from mobius.tasks import MoshiTemporalTask

        config = self._temporal_tiny_config()
        pkg = build_from_module(MoshiTemporalModel(config), config, task=MoshiTemporalTask())
        model = pkg["model"]
        inputs = {inp.name for inp in model.graph.inputs}
        outputs = {out.name for out in model.graph.outputs}
        assert "input_frame" in inputs
        assert "position_ids" in inputs
        assert "hidden" in outputs
        assert "text_logits" in outputs
        assert "present.0.key" in outputs

    def test_temporal_gqa_emits_sliding_window(self):
        """Temporal GQA nodes carry Moshi's sliding window as local_window_size.

        On the GQA (fp16/cuda) path, the temporal transformer's uniform sliding
        window (Moshi ``context``) must reach every GroupQueryAttention node as
        ``local_window_size``.  PersonaPlex deploys this fp16 path, so without it
        long streams would silently run full causal attention.
        """
        import dataclasses

        import onnx_ir as ir

        from mobius.models.moshi import MoshiTemporalModel, moshi_temporal_config
        from mobius.tasks import MoshiTemporalTask

        full_window = moshi_temporal_config().sliding_window
        assert full_window and full_window > 0, "Moshi temporal must be sliding"

        config = dataclasses.replace(self._temporal_tiny_config(), dtype=ir.DataType.FLOAT16)
        pkg = build_from_module(
            MoshiTemporalModel(config),
            config,
            task=MoshiTemporalTask(),
            execution_provider="cuda",
        )
        gqa_nodes = [n for n in pkg["model"].graph if n.op_type == "GroupQueryAttention"]
        assert len(gqa_nodes) == config.num_hidden_layers
        for node in gqa_nodes:
            assert node.attributes["local_window_size"].value == full_window

    def test_depformer_io(self):
        """Depformer model I/O: hidden + prev_token + substep_index -> logits."""
        from mobius.models.moshi import MoshiDepformerModel, moshi_depformer_config
        from mobius.tasks import MoshiDepformerTask

        config = moshi_depformer_config()
        pkg = build_from_module(MoshiDepformerModel(config), config, task=MoshiDepformerTask())
        model = pkg["model"]
        inputs = {inp.name for inp in model.graph.inputs}
        outputs = {out.name for out in model.graph.outputs}
        assert "hidden" in inputs
        assert "prev_token" in inputs
        assert "substep_index" in inputs
        assert "logits" in outputs
        assert "present.0.key" in outputs


class TestBuildCodecGraph:
    """Verify codec tokenizer (Qwen3-TTS-Tokenizer-12Hz) graph construction."""

    @staticmethod
    def _codec_config():
        from mobius._configs import (
            CodecDecoderConfig,
            CodecEncoderConfig,
        )

        return ArchitectureConfig(
            # Use decoder's transformer dims as top-level (from exporter)
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=8,
            intermediate_size=64,
            vocab_size=256,
            max_position_embeddings=128,
            rms_norm_eps=1e-5,
            codec_decoder=CodecDecoderConfig(
                codebook_dim=32,
                codebook_size=64,
                latent_dim=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                head_dim=8,
                rms_norm_eps=1e-5,
                rope_theta=10000.0,
                max_position_embeddings=128,
                decoder_dim=96,
                num_quantizers=4,
                upsample_rates=[2, 2, 2, 2],
                upsampling_ratios=[2, 2],
            ),
            codec_encoder=CodecEncoderConfig(
                codebook_dim=16,
                codebook_size=64,
                hidden_size=32,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                head_dim=8,
                rope_theta=10000.0,
                max_position_embeddings=128,
                num_quantizers=8,
                num_semantic_quantizers=1,
            ),
        )

    def test_package_builds_2_models(self):
        """Build codec tokenizer and verify 2-model package."""
        from mobius.models.qwen3_tts_tokenizer import (
            Qwen3TTSTokenizerV2Model,
        )
        from mobius.tasks import CodecTask

        config = self._codec_config()
        module = Qwen3TTSTokenizerV2Model(config)
        pkg = build_from_module(module, config, task=CodecTask())

        assert "decoder" in pkg
        assert "encoder" in pkg

    def test_decoder_io(self):
        """Verify decoder: codes → waveform."""
        from mobius.models.qwen3_tts_tokenizer import (
            Qwen3TTSTokenizerV2Model,
        )
        from mobius.tasks import CodecTask

        config = self._codec_config()
        module = Qwen3TTSTokenizerV2Model(config)
        pkg = build_from_module(module, config, task=CodecTask())
        decoder = pkg["decoder"]

        input_names = {inp.name for inp in decoder.graph.inputs}
        assert "codes" in input_names

        output_names = {out.name for out in decoder.graph.outputs}
        assert "waveform" in output_names

    def test_encoder_io(self):
        """Verify encoder: waveform → codes."""
        from mobius.models.qwen3_tts_tokenizer import (
            Qwen3TTSTokenizerV2Model,
        )
        from mobius.tasks import CodecTask

        config = self._codec_config()
        module = Qwen3TTSTokenizerV2Model(config)
        pkg = build_from_module(module, config, task=CodecTask())
        encoder = pkg["encoder"]

        input_names = {inp.name for inp in encoder.graph.inputs}
        assert "waveform" in input_names

        output_names = {out.name for out in encoder.graph.outputs}
        assert "codes" in output_names

    def test_registry_lookup(self):
        """Verify qwen3_tts_tokenizer_12hz is registered with codec task."""
        model_cls = registry.get("qwen3_tts_tokenizer_12hz")
        from mobius.models.qwen3_tts_tokenizer import (
            Qwen3TTSTokenizerV2Model,
        )

        assert model_cls is Qwen3TTSTokenizerV2Model
        assert _default_task_for_model("qwen3_tts_tokenizer_12hz") == "codec"


# ===========================================================================
# SSM (Mamba) model tests
# ===========================================================================


class TestBuildMambaGraph:
    """Verify Mamba SSM model builds with SSMCausalLMTask."""

    def _mamba_config(self):
        from mobius._configs import MambaConfig

        return MambaConfig(
            vocab_size=TINY_VOCAB,
            hidden_size=TINY_HIDDEN,
            intermediate_size=TINY_HIDDEN * 2,  # expand=2
            num_hidden_layers=TINY_LAYERS,
            state_size=8,
            conv_kernel=4,
            expand=2,
            time_step_rank=4,
            layer_norm_epsilon=1e-5,
            use_conv_bias=True,
            tie_word_embeddings=True,
        )

    def test_mamba_builds(self):
        """Build Mamba model and verify basic graph structure."""
        from mobius._builder import build_from_module
        from mobius.models.mamba import MambaCausalLMModel
        from mobius.tasks import SSMCausalLMTask

        config = self._mamba_config()
        module = MambaCausalLMModel(config)
        pkg = build_from_module(module, config, task=SSMCausalLMTask())
        model = pkg["model"]

        assert model.graph is not None
        assert len(model.graph.inputs) > 0
        assert len(model.graph.outputs) > 0

    def test_mamba_inputs_no_attention(self):
        """Verify Mamba has input_ids but NOT attention_mask or position_ids."""
        from mobius._builder import build_from_module
        from mobius.models.mamba import MambaCausalLMModel
        from mobius.tasks import SSMCausalLMTask

        config = self._mamba_config()
        module = MambaCausalLMModel(config)
        pkg = build_from_module(module, config, task=SSMCausalLMTask())
        model = pkg["model"]

        input_names = {inp.name for inp in model.graph.inputs}
        assert "input_ids" in input_names
        assert "attention_mask" not in input_names
        assert "position_ids" not in input_names

    def test_mamba_ssm_state_io(self):
        """Verify conv_state + ssm_state per layer in inputs/outputs."""
        from mobius._builder import build_from_module
        from mobius.models.mamba import MambaCausalLMModel
        from mobius.tasks import SSMCausalLMTask

        config = self._mamba_config()
        module = MambaCausalLMModel(config)
        pkg = build_from_module(module, config, task=SSMCausalLMTask())
        model = pkg["model"]

        input_names = {inp.name for inp in model.graph.inputs}
        output_names = {out.name for out in model.graph.outputs}

        for i in range(config.num_hidden_layers):
            assert f"past_states.{i}.conv_state" in input_names
            assert f"past_states.{i}.ssm_state" in input_names
            assert f"present.{i}.conv_state" in output_names
            assert f"present.{i}.ssm_state" in output_names

    def test_mamba_logits_output(self):
        """Verify logits are in the outputs."""
        from mobius._builder import build_from_module
        from mobius.models.mamba import MambaCausalLMModel
        from mobius.tasks import SSMCausalLMTask

        config = self._mamba_config()
        module = MambaCausalLMModel(config)
        pkg = build_from_module(module, config, task=SSMCausalLMTask())
        model = pkg["model"]

        output_names = {out.name for out in model.graph.outputs}
        assert "logits" in output_names

    def test_mamba_has_initializers(self):
        """Verify model has SSM-specific parameters."""
        from mobius._builder import build_from_module
        from mobius.models.mamba import MambaCausalLMModel
        from mobius.tasks import SSMCausalLMTask

        config = self._mamba_config()
        module = MambaCausalLMModel(config)
        pkg = build_from_module(module, config, task=SSMCausalLMTask())
        model = pkg["model"]

        init_names = list(model.graph.initializers)
        assert len(init_names) > 0
        # Check for Mamba-specific parameters
        assert any("embeddings" in n for n in init_names)
        assert any("mixer" in n for n in init_names)
        assert any("norm" in n for n in init_names)

    def test_mamba_registry_lookup(self):
        """Verify 'mamba' is registered and uses SSM task."""
        model_cls = registry.get("mamba")
        from mobius.models.mamba import MambaCausalLMModel

        assert model_cls is MambaCausalLMModel
        assert _default_task_for_model("mamba") == "ssm-text-generation"

    def test_mamba_preprocess_weights_ssm_nesting(self):
        """Verify preprocess_weights maps flat mixer SSM params to nested ssm."""
        import torch

        from mobius.models.mamba import MambaCausalLMModel

        config = self._mamba_config()
        module = MambaCausalLMModel(config)

        # Simulate HF weight names (flat mixer SSM params)
        state_dict = {
            "model.layers.0.mixer.A_log": torch.zeros(1),
            "model.layers.0.mixer.D": torch.zeros(1),
            "model.layers.0.mixer.x_proj.weight": torch.zeros(1),
            "model.layers.0.mixer.dt_proj.weight": torch.zeros(1),
            "model.layers.0.mixer.dt_proj.bias": torch.zeros(1),
            "model.layers.0.mixer.in_proj.weight": torch.zeros(1),
            "model.layers.0.mixer.conv1d.weight": torch.zeros(1),
            "model.layers.0.mixer.out_proj.weight": torch.zeros(1),
            "model.layers.0.norm.weight": torch.zeros(1),
            "model.embeddings.weight": torch.zeros(1),
            "lm_head.weight": torch.zeros(1),
        }
        result = module.preprocess_weights(state_dict)

        # SSM params should be nested under .mixer.ssm.
        assert "model.layers.0.mixer.ssm.A_log" in result
        assert "model.layers.0.mixer.ssm.D" in result
        assert "model.layers.0.mixer.ssm.x_proj.weight" in result
        assert "model.layers.0.mixer.ssm.dt_proj.weight" in result
        assert "model.layers.0.mixer.ssm.dt_proj.bias" in result
        # Non-SSM mixer params stay flat
        assert "model.layers.0.mixer.in_proj.weight" in result
        assert "model.layers.0.mixer.conv1d.weight" in result
        assert "model.layers.0.mixer.out_proj.weight" in result


# ===========================================================================
# FalconMamba SSM model tests
# ===========================================================================


class TestBuildFalconMambaGraph:
    """Verify FalconMamba reuses MambaCausalLMModel via registry."""

    def _falcon_mamba_config(self):
        from mobius._configs import MambaConfig

        return MambaConfig(
            vocab_size=TINY_VOCAB,
            hidden_size=TINY_HIDDEN,
            intermediate_size=TINY_HIDDEN * 2,
            num_hidden_layers=TINY_LAYERS,
            state_size=8,
            conv_kernel=4,
            expand=2,
            time_step_rank=4,
            layer_norm_epsilon=1e-5,
            use_conv_bias=True,
            tie_word_embeddings=True,
        )

    def test_falcon_mamba_registry_lookup(self):
        """Verify 'falcon_mamba' maps to MambaCausalLMModel."""
        model_cls = registry.get("falcon_mamba")
        from mobius.models.mamba import MambaCausalLMModel

        assert model_cls is MambaCausalLMModel
        assert _default_task_for_model("falcon_mamba") == "ssm-text-generation"

    def test_falcon_mamba_builds(self):
        """Build FalconMamba model and verify graph structure."""
        from mobius._builder import build_from_module
        from mobius.models.mamba import MambaCausalLMModel
        from mobius.tasks import SSMCausalLMTask

        config = self._falcon_mamba_config()
        module = MambaCausalLMModel(config)
        pkg = build_from_module(module, config, task=SSMCausalLMTask())
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        output_names = {out.name for out in model.graph.outputs}
        assert "input_ids" in input_names
        assert "logits" in output_names
        # SSM state I/O
        for i in range(config.num_hidden_layers):
            assert f"past_states.{i}.conv_state" in input_names
            assert f"present.{i}.conv_state" in output_names


# ===========================================================================
# Standalone Mamba2/SSD model tests
# ===========================================================================


class TestBuildMamba2Graph:
    """Verify standalone Mamba2/SSD model builds correctly."""

    def _mamba2_config(self):
        from mobius._configs import Mamba2Config

        return Mamba2Config(
            vocab_size=TINY_VOCAB,
            hidden_size=TINY_HIDDEN,
            intermediate_size=TINY_HIDDEN * 2,
            num_hidden_layers=TINY_LAYERS,
            num_heads=8,
            head_dim=16,
            state_size=8,
            n_groups=1,
            conv_kernel=4,
            expand=2,
            layer_norm_epsilon=1e-5,
            use_conv_bias=True,
        )

    def test_mamba2_registry_lookup(self):
        """Verify 'mamba2' maps to Mamba2CausalLMModel."""
        model_cls = registry.get("mamba2")
        from mobius.models.mamba import Mamba2CausalLMModel

        assert model_cls is Mamba2CausalLMModel
        assert _default_task_for_model("mamba2") == "ssm2-text-generation"

    def test_mamba2_builds(self):
        """Build standalone Mamba2 model and verify graph structure."""
        from mobius._builder import build_from_module
        from mobius.models.mamba import Mamba2CausalLMModel
        from mobius.tasks import SSM2CausalLMTask

        config = self._mamba2_config()
        module = Mamba2CausalLMModel(config)
        pkg = build_from_module(module, config, task=SSM2CausalLMTask())
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        output_names = {out.name for out in model.graph.outputs}
        assert "input_ids" in input_names
        assert "logits" in output_names

    def test_mamba2_state_io(self):
        """Verify 4D SSM state shapes in graph I/O."""
        from mobius._builder import build_from_module
        from mobius.models.mamba import Mamba2CausalLMModel
        from mobius.tasks import SSM2CausalLMTask

        config = self._mamba2_config()
        module = Mamba2CausalLMModel(config)
        pkg = build_from_module(module, config, task=SSM2CausalLMTask())
        model = pkg["model"]

        input_names = {inp.name for inp in model.graph.inputs}
        output_names = {out.name for out in model.graph.outputs}

        # Every layer should have conv_state + ssm_state
        for i in range(config.num_hidden_layers):
            assert f"past_states.{i}.conv_state" in input_names
            assert f"past_states.{i}.ssm_state" in input_names
            assert f"present.{i}.conv_state" in output_names
            assert f"present.{i}.ssm_state" in output_names

    def test_mamba2_preprocess_weights(self):
        """Verify SSM params stay at mixer level (no nesting)."""
        from mobius.models.mamba import Mamba2CausalLMModel

        config = self._mamba2_config()
        module = Mamba2CausalLMModel(config)

        import torch

        state_dict = {
            "backbone.layers.0.mixer.A_log": torch.zeros(8),
            "backbone.layers.0.mixer.D": torch.zeros(8),
            "backbone.layers.0.mixer.dt_bias": torch.zeros(8),
            "backbone.layers.0.mixer.in_proj.weight": torch.zeros(280, 64),
            "backbone.layers.0.norm.weight": torch.zeros(64),
        }
        result = module.preprocess_weights(state_dict)

        # SSM params stay directly on mixer (no .ssm. nesting)
        assert "backbone.layers.0.mixer.A_log" in result
        assert "backbone.layers.0.mixer.D" in result
        assert "backbone.layers.0.mixer.dt_bias" in result
        # Non-SSM params stay as-is
        assert "backbone.layers.0.mixer.in_proj.weight" in result
        assert "backbone.layers.0.norm.weight" in result


# ===========================================================================
# Hybrid Mamba2+Attention (Bamba) model tests
# ===========================================================================


class TestBuildBambaGraph:
    """Verify Bamba hybrid Mamba2+Attention model builds correctly."""

    def _bamba_config(self):
        from mobius._configs import BambaConfig

        return BambaConfig(
            vocab_size=TINY_VOCAB,
            hidden_size=TINY_HIDDEN,
            intermediate_size=TINY_INTERMEDIATE,
            num_hidden_layers=4,
            num_attention_heads=TINY_HEADS,
            num_key_value_heads=TINY_KV_HEADS,
            rms_norm_eps=1e-5,
            layer_types=[
                "mamba2",
                "full_attention",
                "mamba2",
                "mamba2",
            ],
            mamba_n_heads=4,
            mamba_d_head=32,
            mamba_d_state=8,
            mamba_n_groups=1,
            mamba_d_conv=4,
            mamba_expand=2,
            mamba_conv_bias=True,
            mamba_proj_bias=False,
            hidden_act="silu",
            head_dim=TINY_HIDDEN // TINY_HEADS,
            # Bamba's attention layers use standard RoPE; enable it
            # explicitly since ArchitectureConfig now defaults ``rope_type``
            # to ``None`` (NoPE) to represent "no RoPE" structurally.
            rope_type="default",
        )

    def test_bamba_builds(self):
        """Build Bamba model and verify basic graph structure."""
        from mobius._builder import build_from_module
        from mobius.models.bamba import BambaCausalLMModel
        from mobius.tasks import HybridCausalLMTask

        config = self._bamba_config()
        module = BambaCausalLMModel(config)
        pkg = build_from_module(module, config, task=HybridCausalLMTask())
        model = pkg["model"]

        assert model.graph is not None
        assert len(model.graph.inputs) > 0
        assert len(model.graph.outputs) > 0

    def test_bamba_hybrid_cache_io(self):
        """Verify mixed Mamba2/attention cache I/O."""
        from mobius._builder import build_from_module
        from mobius.models.bamba import BambaCausalLMModel
        from mobius.tasks import HybridCausalLMTask

        config = self._bamba_config()
        module = BambaCausalLMModel(config)
        pkg = build_from_module(module, config, task=HybridCausalLMTask())
        model = pkg["model"]

        input_names = {inp.name for inp in model.graph.inputs}
        output_names = {out.name for out in model.graph.outputs}

        assert "past_key_values.0.conv_state" in input_names
        assert "past_key_values.0.ssm_state" in input_names
        assert "present.0.conv_state" in output_names
        assert "present.0.ssm_state" in output_names
        assert "past_key_values.1.key" in input_names
        assert "past_key_values.1.value" in input_names
        assert "present.1.key" in output_names
        assert "present.1.value" in output_names
        assert "past_key_values.2.conv_state" in input_names
        assert "past_key_values.3.conv_state" in input_names

    def test_bamba_registry_lookup(self):
        """Verify bamba is registered and uses hybrid task."""
        model_cls = registry.get("bamba")
        from mobius.models.bamba import BambaCausalLMModel

        assert model_cls is BambaCausalLMModel
        assert _default_task_for_model("bamba") == "hybrid-text-generation"

    def test_bamba_preprocess_weights(self):
        """Verify preprocess_weights passes SSM params through unchanged."""
        import torch

        from mobius.models.bamba import BambaCausalLMModel

        config = self._bamba_config()
        module = BambaCausalLMModel(config)

        state_dict = {
            "model.layers.0.mamba.A_log": torch.zeros(4),
            "model.layers.0.mamba.D": torch.zeros(4),
            "model.layers.0.mamba.dt_bias": torch.zeros(4),
            "model.layers.0.mamba.in_proj.weight": torch.zeros(1),
            "model.layers.0.mamba.conv1d.weight": torch.zeros(1),
            "model.layers.0.mamba.norm.weight": torch.zeros(1),
            "model.layers.0.mamba.out_proj.weight": torch.zeros(1),
            "model.layers.1.self_attn.q_proj.weight": torch.zeros(1),
            "model.embed_tokens.weight": torch.zeros(1),
            "lm_head.weight": torch.zeros(1),
        }
        result = module.preprocess_weights(state_dict)

        # SSM params stay directly on mamba (no .ssm. nesting)
        assert "model.layers.0.mamba.A_log" in result
        assert "model.layers.0.mamba.D" in result
        assert "model.layers.0.mamba.dt_bias" in result
        assert "model.layers.0.mamba.in_proj.weight" in result
        assert "model.layers.1.self_attn.q_proj.weight" in result


# ===========================================================================
# Hybrid Mamba2+Attention+MLP (NemotronH) model tests
# ===========================================================================


class TestBuildNemotronHGraph:
    """Verify NemotronH hybrid model weight renaming."""

    def _nemotron_h_config(self):
        from mobius._configs import NemotronHConfig

        # 4 layers: mamba2, mlp, full_attention, mlp
        return NemotronHConfig(
            vocab_size=TINY_VOCAB,
            hidden_size=TINY_HIDDEN,
            intermediate_size=TINY_INTERMEDIATE,
            num_hidden_layers=4,
            num_attention_heads=TINY_HEADS,
            num_key_value_heads=TINY_KV_HEADS,
            rms_norm_eps=1e-5,
            layer_types=["mamba2", "mlp", "full_attention", "mlp"],
            mamba_n_heads=TINY_KV_HEADS,
            mamba_d_head=TINY_HEAD_DIM,
            mamba_d_state=16,
            mamba_n_groups=1,
            mamba_d_conv=4,
            mamba_expand=2,
            hidden_act="relu2",
            head_dim=TINY_HEAD_DIM,
        )

    def test_nemotron_h_preprocess_weights(self):
        """Verify preprocess_weights routes by layer type and nests SSM params."""
        import torch

        from mobius.models.nemotron_h import NemotronHCausalLMModel

        config = self._nemotron_h_config()
        module = NemotronHCausalLMModel(config)

        # Simulate HF NemotronH weight names (backbone.* prefix,
        # all layer mixers named "mixer.*" regardless of type)
        state_dict = {
            # Embeddings & final norm
            "backbone.embeddings.weight": torch.zeros(1),
            "backbone.norm_f.weight": torch.zeros(1),
            "lm_head.weight": torch.zeros(1),
            # Layer 0: mamba2 — SSM params + mixer params
            "backbone.layers.0.norm.weight": torch.zeros(1),
            "backbone.layers.0.mixer.A_log": torch.zeros(4),
            "backbone.layers.0.mixer.D": torch.zeros(4),
            "backbone.layers.0.mixer.dt_bias": torch.zeros(4),
            "backbone.layers.0.mixer.in_proj.weight": torch.zeros(1),
            "backbone.layers.0.mixer.conv1d.weight": torch.zeros(1),
            "backbone.layers.0.mixer.out_proj.weight": torch.zeros(1),
            "backbone.layers.0.mixer.norm.weight": torch.zeros(1),
            # Layer 1: mlp
            "backbone.layers.1.norm.weight": torch.zeros(1),
            "backbone.layers.1.mixer.up_proj.weight": torch.zeros(1),
            "backbone.layers.1.mixer.down_proj.weight": torch.zeros(1),
            # Layer 2: full_attention
            "backbone.layers.2.norm.weight": torch.zeros(1),
            "backbone.layers.2.mixer.q_proj.weight": torch.zeros(1),
            "backbone.layers.2.mixer.k_proj.weight": torch.zeros(1),
            "backbone.layers.2.mixer.v_proj.weight": torch.zeros(1),
            "backbone.layers.2.mixer.o_proj.weight": torch.zeros(1),
            # Layer 3: mlp
            "backbone.layers.3.norm.weight": torch.zeros(1),
            "backbone.layers.3.mixer.up_proj.weight": torch.zeros(1),
            "backbone.layers.3.mixer.down_proj.weight": torch.zeros(1),
        }
        result = module.preprocess_weights(state_dict)

        # Global renames: backbone.embeddings -> model.embed_tokens,
        # backbone.norm_f -> model.norm
        assert "model.embed_tokens.weight" in result
        assert "model.norm.weight" in result

        # Layer 0 (mamba2): SSM params directly under mamba (no nesting)
        assert "model.layers.0.mamba.A_log" in result
        assert "model.layers.0.mamba.D" in result
        assert "model.layers.0.mamba.dt_bias" in result
        # Non-SSM mamba params stay under mamba.*
        assert "model.layers.0.mamba.in_proj.weight" in result
        assert "model.layers.0.mamba.conv1d.weight" in result
        assert "model.layers.0.mamba.out_proj.weight" in result
        assert "model.layers.0.mamba.norm.weight" in result

        # Layer 1 (mlp): mixer -> mlp
        assert "model.layers.1.mlp.up_proj.weight" in result
        assert "model.layers.1.mlp.down_proj.weight" in result

        # Layer 2 (full_attention): mixer -> self_attn
        assert "model.layers.2.self_attn.q_proj.weight" in result
        assert "model.layers.2.self_attn.k_proj.weight" in result
        assert "model.layers.2.self_attn.v_proj.weight" in result
        assert "model.layers.2.self_attn.o_proj.weight" in result

        # Layer 3 (mlp): mixer -> mlp
        assert "model.layers.3.mlp.up_proj.weight" in result
        assert "model.layers.3.mlp.down_proj.weight" in result

        # Per-layer norms keep their names
        assert "model.layers.0.norm.weight" in result
        assert "model.layers.2.norm.weight" in result

        # No original backbone.* keys should remain
        for key in result:
            assert not key.startswith("backbone."), f"Unrenamed key: {key}"

    def test_nemotron_h_moe_preprocess_weights(self):
        """Verify stacked 3D MoE expert tensors are split into per-expert 2D weights."""
        import torch

        from mobius._configs import NemotronHConfig
        from mobius.models.nemotron_h import NemotronHCausalLMModel

        config = NemotronHConfig(
            vocab_size=TINY_VOCAB,
            hidden_size=TINY_HIDDEN,
            intermediate_size=TINY_INTERMEDIATE,
            num_hidden_layers=2,
            num_attention_heads=TINY_HEADS,
            num_key_value_heads=TINY_KV_HEADS,
            rms_norm_eps=1e-5,
            layer_types=["full_attention", "moe"],
            mamba_n_heads=TINY_KV_HEADS,
            mamba_d_head=TINY_HEAD_DIM,
            mamba_d_state=16,
            mamba_n_groups=1,
            mamba_d_conv=4,
            mamba_expand=2,
            hidden_act="relu2",
            head_dim=TINY_HEAD_DIM,
            num_local_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=TINY_INTERMEDIATE,
        )
        module = NemotronHCausalLMModel(config)

        num_experts = 4
        # Stacked 3D expert tensors (HF format): (num_experts, out_dim, in_dim)
        up_proj_stacked = torch.randn(num_experts, TINY_INTERMEDIATE, TINY_HIDDEN)
        down_proj_stacked = torch.randn(num_experts, TINY_HIDDEN, TINY_INTERMEDIATE)

        state_dict = {
            # Embeddings & norm
            "backbone.embeddings.weight": torch.zeros(1),
            "backbone.norm_f.weight": torch.zeros(1),
            "lm_head.weight": torch.zeros(1),
            # Layer 0: full_attention
            "backbone.layers.0.norm.weight": torch.zeros(1),
            "backbone.layers.0.mixer.q_proj.weight": torch.zeros(1),
            "backbone.layers.0.mixer.k_proj.weight": torch.zeros(1),
            "backbone.layers.0.mixer.v_proj.weight": torch.zeros(1),
            "backbone.layers.0.mixer.o_proj.weight": torch.zeros(1),
            # Layer 1: moe — stacked expert weights (3D)
            "backbone.layers.1.norm.weight": torch.zeros(1),
            "backbone.layers.1.mixer.experts.up_proj": up_proj_stacked,
            "backbone.layers.1.mixer.experts.down_proj": down_proj_stacked,
            # MoE gate
            "backbone.layers.1.mixer.gate.weight": torch.zeros(1),
            "backbone.layers.1.mixer.gate.e_score_correction_bias": torch.zeros(1),
            # MoE shared experts
            "backbone.layers.1.mixer.shared_experts.up_proj.weight": torch.zeros(1),
            "backbone.layers.1.mixer.shared_experts.down_proj.weight": torch.zeros(1),
        }

        result = module.preprocess_weights(state_dict)

        # Stacked expert keys must NOT be in the result
        assert "model.layers.1.moe.experts.up_proj" not in result
        assert "model.layers.1.moe.experts.down_proj" not in result

        # Per-expert keys must exist with correct shapes
        for i in range(num_experts):
            up_key = f"model.layers.1.moe.experts.{i}.up_proj.weight"
            down_key = f"model.layers.1.moe.experts.{i}.down_proj.weight"
            assert up_key in result, f"Missing {up_key}"
            assert down_key in result, f"Missing {down_key}"
            assert result[up_key].shape == (TINY_INTERMEDIATE, TINY_HIDDEN), (
                f"{up_key} shape {result[up_key].shape}"
            )
            assert result[down_key].shape == (TINY_HIDDEN, TINY_INTERMEDIATE), (
                f"{down_key} shape {result[down_key].shape}"
            )
            # Verify the data matches the original slice
            torch.testing.assert_close(result[up_key], up_proj_stacked[i])
            torch.testing.assert_close(result[down_key], down_proj_stacked[i])

        # Gate weights are renamed correctly
        assert "model.layers.1.moe.gate.weight" in result
        assert "model.layers.1.moe.gate.e_score_correction_bias" in result

        # Shared expert weights are renamed correctly
        assert "model.layers.1.moe.shared_experts.up_proj.weight" in result
        assert "model.layers.1.moe.shared_experts.down_proj.weight" in result

        # No original backbone.* keys should remain
        for key in result:
            assert not key.startswith("backbone."), f"Unrenamed key: {key}"


# ===========================================================================
# Hybrid SSM+Attention (Jamba) model tests
# ===========================================================================


class TestBuildJambaGraph:
    """Verify Jamba hybrid SSM+Attention model builds correctly."""

    def _jamba_config(self):
        from mobius._configs import JambaConfig

        # 4 layers: attn_layer_period=2, attn_layer_offset=1
        # → layer 0: mamba, 1: attention, 2: mamba, 3: attention
        # expert_layer_period=2, expert_layer_offset=1
        # → layer 0: dense, 1: MoE, 2: dense, 3: MoE
        return JambaConfig(
            vocab_size=TINY_VOCAB,
            hidden_size=TINY_HIDDEN,
            intermediate_size=TINY_INTERMEDIATE,
            num_hidden_layers=4,
            num_attention_heads=TINY_HEADS,
            num_key_value_heads=TINY_KV_HEADS,
            head_dim=TINY_HIDDEN // TINY_HEADS,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            layer_types=["mamba", "full_attention", "mamba", "full_attention"],
            mamba_d_state=8,
            mamba_d_conv=4,
            mamba_expand=2,
            mamba_dt_rank=4,
            attn_layer_period=2,
            attn_layer_offset=1,
            expert_layer_period=2,
            expert_layer_offset=1,
            num_local_experts=2,
            num_experts_per_tok=1,
            # Jamba's attention layers use standard RoPE; enable it
            # explicitly since ArchitectureConfig defaults ``rope_type`` to
            # ``None`` (NoPE) to express "no RoPE" structurally.
            rope_type="default",
        )

    def test_jamba_builds(self):
        """Build Jamba model and verify basic graph structure."""
        from mobius._builder import build_from_module
        from mobius.models.jamba import JambaCausalLMModel
        from mobius.tasks import HybridCausalLMTask

        config = self._jamba_config()
        module = JambaCausalLMModel(config)
        pkg = build_from_module(module, config, task=HybridCausalLMTask())
        model = pkg["model"]

        assert model.graph is not None
        assert len(model.graph.inputs) > 0

    def test_jamba_has_logits_output(self):
        """Jamba model should produce logits output."""
        from mobius._builder import build_from_module
        from mobius.models.jamba import JambaCausalLMModel
        from mobius.tasks import HybridCausalLMTask

        config = self._jamba_config()
        module = JambaCausalLMModel(config)
        pkg = build_from_module(module, config, task=HybridCausalLMTask())
        model = pkg["model"]

        output_names = [o.name for o in model.graph.outputs]
        assert "logits" in output_names

    def test_jamba_hybrid_cache_io(self):
        """Verify Jamba has mixed cache outputs (mamba + attention)."""
        from mobius._builder import build_from_module
        from mobius.models.jamba import JambaCausalLMModel
        from mobius.tasks import HybridCausalLMTask

        config = self._jamba_config()
        module = JambaCausalLMModel(config)
        pkg = build_from_module(module, config, task=HybridCausalLMTask())
        model = pkg["model"]

        output_names = [o.name for o in model.graph.outputs]
        # Mamba layers (0, 2): conv_state + ssm_state
        assert "present.0.conv_state" in output_names
        assert "present.0.ssm_state" in output_names
        assert "present.2.conv_state" in output_names
        assert "present.2.ssm_state" in output_names
        # Attention layers (1, 3): key + value
        assert "present.1.key" in output_names
        assert "present.1.value" in output_names
        assert "present.3.key" in output_names
        assert "present.3.value" in output_names

    def test_jamba_registry_lookup(self):
        """Jamba should be in the registry as JambaCausalLMModel."""
        from mobius._registry import registry

        model_cls = registry.get("jamba")
        assert model_cls.__name__ == "JambaCausalLMModel"

    def test_jamba_preprocess_weights_moe_renames(self):
        """Verify MoE expert weight renames and SSM nesting."""
        import torch

        from mobius.models.jamba import JambaCausalLMModel

        config = self._jamba_config()
        module = JambaCausalLMModel(config)

        state_dict = {
            # SSM param: should nest under mamba.ssm
            "model.layers.0.mamba.A_log": torch.zeros(1),
            "model.layers.0.mamba.D": torch.zeros(1),
            "model.layers.0.mamba.dt_layernorm.weight": torch.zeros(1),
            # MoE router → gate
            "model.layers.1.feed_forward.router.weight": torch.zeros(1),
            # Non-mamba params pass through
            "model.layers.0.mamba.in_proj.weight": torch.zeros(1),
            "lm_head.weight": torch.zeros(1),
        }
        result = module.preprocess_weights(state_dict)

        # SSM params nested
        assert "model.layers.0.mamba.ssm.A_log" in result
        assert "model.layers.0.mamba.ssm.D" in result
        assert "model.layers.0.mamba.ssm.dt_layernorm.weight" in result
        # MoE gate renamed
        assert "model.layers.1.feed_forward.gate.weight" in result
        # Non-SSM stays flat
        assert "model.layers.0.mamba.in_proj.weight" in result


# ===========================================================================
# Registry completeness
# ===========================================================================

# Model types exercised by non-parametrized test classes above (VLM,
# whisper, audio, TTS, diffusion, etc.).  Keep sorted for readability.
_SPECIALIZED_TEST_MODEL_TYPES: set[str] = {
    # LLaDA masked-diffusion LM (co-located src/mobius/models/llada_test.py):
    # bidirectional Llama backbone with a masked-diffusion task, so it has no
    # attention_mask / KV cache and does not fit the generic causal-LM harness.
    "llada",
    # VLM alias tests (test_llava_aliases_build)
    "aya_vision",
    "chameleon",
    "cohere2_vision",
    "deepseek_vl",
    "deepseek_vl_hybrid",
    # FastConformer-RNNT (co-located src/mobius/models/nemo_rnnt_test.py)
    "fastconformer_rnnt",
    "florence2",
    "fuyu",
    "glm4v",
    "glm4v_moe",
    "got_ocr2",
    "idefics2",
    "idefics3",
    "instructblip",
    "instructblipvideo",
    "internvl",
    "internvl_chat",
    "internvl2",
    "janus",
    "llava_next",
    "llava_next_video",
    "llava_onevision",
    "mistral3",
    "molmo",
    "ovis2",
    "paligemma",
    "pixtral",
    "smolvlm",
    "video_llava",
    "vipllava",
    # VLM dedicated tests
    "blip-2",
    "deepseek_vl_v2",
    "gemma3",
    "gemma4",
    "gemma4_unified",
    "gemma4_unified_text",
    "llava",
    "mllama",
    "phi3_v",
    "phi4-siglip",
    "phi4_multimodal",
    "phi4mm",
    "qwen2_5_vl",
    "qwen2_5_vl_text",
    "qwen2_vl",
    "qwen2_vl_text",
    "qwen3_5",
    "qwen3_5_vl",
    "qwen3_vl",
    "qwen3_vl_single",
    # Audio alias tests (test_audio_aliases_build)
    "data2vec-audio",
    "hubert",
    "mctct",
    "musicgen",
    "seamless_m4t",
    "seamless_m4t_v2",
    "sew",
    "sew-d",
    "sortformer",
    "speecht5",
    "unispeech",
    "unispeech-sat",
    "voxtral_encoder",
    "wav2vec2",
    "wav2vec2-bert",
    "wav2vec2-conformer",
    "wavlm",
    # Audio/TTS dedicated tests
    "fun_asr",
    "mms",
    "qwen3_asr",
    "qwen3_forced_aligner",
    "qwen3_tts",
    "qwen3_tts_tokenizer_12hz",
    "whisper",
    # SSM dedicated tests
    "falcon_mamba",
    "mamba",
    "mamba2",
    # Hybrid SSM+Attention dedicated tests
    "bamba",
    "jamba",
    # Speculative-decoding draft models with bespoke IO contracts
    # (DFlash drafter takes noise_embedding + target_hidden instead of
    # input_ids; the generic ALL_CAUSAL_LM_CONFIGS matrix can't drive it).
    # Covered by src/mobius/models/_dflash_test.py.
    "DFlashDraftModel",
    # Gemma4-Assistant: bespoke IO contract (consumes inputs_embeds +
    # the target's shared KV instead of input_ids), so the generic
    # ALL_CAUSAL_LM_CONFIGS matrix can't drive it. Covered by
    # src/mobius/models/_gemma4_assistant_test.py.
    "gemma4_assistant",
    "Gemma4AssistantForCausalLM",
    "gemma4_unified_assistant",
    "Gemma4UnifiedAssistantForCausalLM",
    # Qwen3.6 MTP self-speculative head: bespoke IO contract (consumes
    # inputs_embeds + the target's hidden_states instead of input_ids;
    # borrows the target's embed/lm_head), so the generic
    # ALL_CAUSAL_LM_CONFIGS matrix can't drive it. Covered by
    # src/mobius/models/_qwen35_mtp_test.py.
    "Qwen35MtpModel",
    # EAGLE-3 drafter: bespoke IO contract (inputs_embeds, fused_hidden,
    # recycled_hidden and draft-vocab logits). Covered by _eagle3_test.py.
    "Eagle3LlamaForCausalLM",
    "LlamaForCausalLMEagle3",
    "Eagle3Speculator",
    "Eagle3DraftModel",
}

# Registered model types that truly have no test coverage yet.
# This set should be empty or near-empty. If a NEW model is registered
# Internal aliases whose real HF counterpart is already tested.
# These are registered in the registry for production use but removed
# from test configs because they duplicate existing coverage or
# cannot be tested with our generic test infrastructure.
_KNOWN_UNTESTED_MODEL_TYPES: set[str] = {
    "deepseek_v2_moe",  # Alias for deepseek_v2 — tested via deepseek_v2
    "qwen3_5_vl_text",  # VL text decoder — tested via parent VL model
}


class TestRegistryCompleteness:
    """Ensure every registered model type has a test config entry."""

    def test_all_registered_models_have_test_coverage(self):
        """Every model_type in the registry must be accounted for.

        A model_type is *covered* if it appears in parametrized test
        configs, auto-generated configs, a specialized test class, or
        the known-untested allowlist.  New registrations that aren't
        covered anywhere will cause this test to fail.
        """
        all_covered = (
            {mt for mt, _, _ in ALL_CONFIGS}
            | {mt for mt, _, _ in AUTO_GENERATED_CONFIGS}
            | _SPECIALIZED_TEST_MODEL_TYPES
            | _KNOWN_UNTESTED_MODEL_TYPES
        )
        registered = set(registry.architectures())
        missing = registered - all_covered
        assert not missing, (
            f"Registered model types without test coverage: "
            f"{sorted(missing)}. Add a test config to "
            "tests/_test_configs.py, a specialized test class, or "
            "acknowledge in _KNOWN_UNTESTED_MODEL_TYPES."
        )

    def test_known_untested_is_minimal(self):
        """Entries in _KNOWN_UNTESTED should still be registered.

        If a model_type is removed from the registry, it should also
        be removed from _KNOWN_UNTESTED_MODEL_TYPES.  If a test is
        added for it, it should move to the appropriate config list or
        _SPECIALIZED_TEST_MODEL_TYPES.
        """
        registered = set(registry.architectures())
        stale = _KNOWN_UNTESTED_MODEL_TYPES - registered
        assert not stale, (
            f"Entries in _KNOWN_UNTESTED_MODEL_TYPES that are no longer "
            f"registered: {sorted(stale)}. Remove them."
        )


class TestBuildStaticCacheGraph:
    """Verify CausalLMTask(static_cache=True) builds a valid graph."""

    MAX_SEQ_LEN = 128

    def _build_static_cache_model(self, model_type: str = "qwen2", **config_overrides):
        """Build a model with CausalLMTask(static_cache=True) and return (model, config)."""
        from mobius.tasks import CausalLMTask

        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = CausalLMTask(static_cache=True, max_seq_len=self.MAX_SEQ_LEN)
        pkg = task.build(module, config)
        return pkg["model"], config

    def test_static_cache_graph_builds(self):
        """Build a Qwen2 model with static cache."""
        model, _ = self._build_static_cache_model()

        assert model.graph is not None
        assert len(model.graph.inputs) > 0
        assert len(model.graph.outputs) > 0

    def test_static_cache_graph_inputs(self):
        """Verify expected inputs: standard + per-layer caches + shared."""
        model, config = self._build_static_cache_model()
        input_names = {inp.name for inp in model.graph.inputs}
        num_layers = config.num_hidden_layers

        # Standard inputs
        assert "input_ids" in input_names
        assert "position_ids" in input_names

        # No attention_mask in static cache mode — causal masking is
        # handled by is_causal=1 on the Attention op.
        assert "attention_mask" not in input_names

        # Per-layer static cache inputs
        for i in range(num_layers):
            assert f"key_cache.{i}" in input_names, f"Missing key_cache.{i}"
            assert f"value_cache.{i}" in input_names, f"Missing value_cache.{i}"

        # Shared cache management inputs
        assert "write_indices" in input_names
        assert "nonpad_kv_seqlen" in input_names

        # Exact count: 2 standard + 2*num_layers caches + 2 shared
        expected_count = 2 + 2 * num_layers + 2
        assert len(model.graph.inputs) == expected_count, (
            f"Expected {expected_count} inputs, got {len(model.graph.inputs)}"
        )

    def test_static_cache_graph_outputs(self):
        """Verify outputs: logits + updated caches per layer."""
        model, config = self._build_static_cache_model()
        output_names = {out.name for out in model.graph.outputs}
        num_layers = config.num_hidden_layers

        assert "logits" in output_names

        # Updated caches per layer (not present.{i}.key/value)
        for i in range(num_layers):
            assert f"updated_key_cache.{i}" in output_names, f"Missing updated_key_cache.{i}"
            assert f"updated_value_cache.{i}" in output_names, (
                f"Missing updated_value_cache.{i}"
            )

        # Should NOT have dynamic cache outputs
        assert not any(n.startswith("present.") for n in output_names), (
            "Static cache graph should not have present.* outputs"
        )

        # Exact count: 1 logits + 2*num_layers updated caches
        expected_count = 1 + 2 * num_layers
        assert len(model.graph.outputs) == expected_count, (
            f"Expected {expected_count} outputs, got {len(model.graph.outputs)}"
        )

    def test_static_cache_has_tensorscatter_and_attention(self):
        """Verify graph contains TensorScatter and Attention ops."""
        model, _ = self._build_static_cache_model()

        op_types = {n.op_type for n in model.graph}
        assert "TensorScatter" in op_types, "Static cache graph should use TensorScatter"
        assert "Attention" in op_types, "Static cache graph should use Attention"

    def test_static_cache_has_initializers(self):
        """Verify the graph has model parameters."""
        model, _ = self._build_static_cache_model()

        init_names = list(model.graph.initializers)
        assert len(init_names) > 0
        assert any("embed_tokens" in n for n in init_names)
        assert any("self_attn" in n for n in init_names)
        assert any("mlp" in n for n in init_names)

    def test_static_cache_graph_validates(self):
        """Verify the graph survives a serialization round-trip."""
        model, _config = self._build_static_cache_model()
        proto = ir.serde.serialize_model(model)
        assert len(proto.SerializeToString()) > 0

    def test_static_cache_attention_uses_maskless_causal_alignment(self):
        """Verify maskless external-cache Attention uses built-in causality."""
        model, config = self._build_static_cache_model()

        attention_nodes = [n for n in model.graph if n.op_type == "Attention"]
        assert len(attention_nodes) == config.num_hidden_layers

        for node in attention_nodes:
            is_causal = node.attributes.get("is_causal")
            assert is_causal is not None, (
                f"Attention node {node.name} missing is_causal attribute"
            )
            assert is_causal.as_int() == 1, (
                f"Attention node {node.name} should have is_causal=1"
            )

    def test_static_cache_attention_no_attn_mask_input(self):
        """Verify Attention ops do NOT receive attn_mask in static cache mode."""
        model, config = self._build_static_cache_model()

        attention_nodes = [n for n in model.graph if n.op_type == "Attention"]
        assert len(attention_nodes) == config.num_hidden_layers

        for node in attention_nodes:
            # Input 3 (0-indexed) is attn_mask — should be empty/None
            attn_mask_input = node.inputs[3]
            assert attn_mask_input is None or attn_mask_input.name == "", (
                f"Attention node {node.name} should not have attn_mask "
                f"connected, but got input: {attn_mask_input}"
            )

    def test_static_cache_moe_graph_builds(self):
        """Build a MoE model (qwen2_moe) with static cache."""
        model, _config = self._build_static_cache_model(
            model_type="qwen2_moe",
            num_local_experts=4,
            num_experts_per_tok=2,
            attn_qkv_bias=True,
            shared_expert_intermediate_size=64,
        )

        assert model.graph is not None
        assert len(model.graph.inputs) > 0
        assert len(model.graph.outputs) > 0

        input_names = {inp.name for inp in model.graph.inputs}
        assert "input_ids" in input_names
        assert "position_ids" in input_names
        assert "attention_mask" not in input_names

        # Verify TensorScatter and Attention ops are present
        op_types = {n.op_type for n in model.graph}
        assert "TensorScatter" in op_types
        assert "Attention" in op_types

    def test_outputs_have_shapes_and_dtypes(self):
        """Verify shape inference populates all output shapes and dtypes."""
        model, _ = self._build_static_cache_model()
        _assert_outputs_have_shapes_and_dtypes({"model": model}, "qwen2-static")

    def _build_gemma4_static(self):
        """Build gemma4_text with static cache: KV-sharing + dual head_dim."""
        from mobius._configs import Gemma4Config
        from mobius.tasks import CausalLMTask

        config = _base_config(
            _config_cls=Gemma4Config,
            num_hidden_layers=6,
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
            num_kv_shared_layers=2,  # first_kv_shared = 4 -> layers 0..3 own a cache
            sliding_window=8,
            global_head_dim=2 * TINY_HEAD_DIM,
            global_rope_theta=10_000.0,
            rope_local_base_freq=10_000.0,
            attn_qk_norm=True,
            hidden_size_per_layer_input=0,
        )
        module = registry.get("gemma4_text")(config)
        pkg = CausalLMTask(static_cache=True, max_seq_len=self.MAX_SEQ_LEN).build(
            module, config
        )
        return pkg["model"], config

    def test_gemma4_static_cache_only_for_cache_owning_layers(self):
        """Gemma4 KV-shared layers own no cache: 4 cache buffers, not 6."""
        model, _ = self._build_gemma4_static()
        input_names = {inp.name for inp in model.graph.inputs}
        # first_kv_shared_layer_idx = 6 - 2 = 4 -> cache-owning layers 0..3
        for i in range(4):
            assert f"key_cache.{i}" in input_names
            assert f"value_cache.{i}" in input_names
        # No cache for the KV-shared layers (indices 4, 5).
        assert "key_cache.4" not in input_names
        assert "key_cache.5" not in input_names
        output_names = {out.name for out in model.graph.outputs}
        for i in range(4):
            assert f"updated_key_cache.{i}" in output_names
        assert "updated_key_cache.4" not in output_names

    def test_gemma4_static_cache_dual_head_dim(self):
        """Sliding (head_dim) and full (global_head_dim) cache buffers differ."""
        model, _ = self._build_gemma4_static()
        by_name = {inp.name: inp for inp in model.graph.inputs}
        # num_kv_heads=TINY_KV_HEADS; sliding head_dim=TINY_HEAD_DIM,
        # full head_dim=2*TINY_HEAD_DIM. Cache-owning layer 2 is full_attention.
        sliding_kv = int(by_name["key_cache.0"].shape[2])  # layer 0 sliding
        full_kv = int(by_name["key_cache.2"].shape[2])  # layer 2 full
        assert sliding_kv == TINY_KV_HEADS * TINY_HEAD_DIM
        assert full_kv == TINY_KV_HEADS * (2 * TINY_HEAD_DIM)

    def test_gemma4_static_cache_has_tensorscatter(self):
        """Gemma4 static cache uses TensorScatter for in-place KV writes."""
        model, _ = self._build_gemma4_static()
        op_types = {n.op_type for n in model.graph}
        assert "TensorScatter" in op_types

    def test_gemma4_static_qnn_lowering_is_htp_friendly(self):
        """The qnn build lowers all ops the QNN HTP backend cannot run.

        RotaryEmbedding -> rotate-half, TensorScatter -> ScatterND, Tile ->
        Expand, Range -> Constant, Attention -> SDPA are all HTP-unsupported and
        must be gone; their HTP-friendly replacements must appear.
        """
        from mobius._builder import build_from_module
        from mobius._configs import Gemma4Config
        from mobius.tasks import CausalLMTask

        config = _base_config(
            _config_cls=Gemma4Config,
            num_hidden_layers=3,
            layer_types=["sliding_attention", "full_attention", "sliding_attention"],
            num_kv_shared_layers=1,
            sliding_window=8,
            global_head_dim=2 * TINY_HEAD_DIM,
            global_rope_theta=10_000.0,
            rope_local_base_freq=10_000.0,
            attn_qk_norm=True,
            hidden_size_per_layer_input=0,
        )
        module = registry.get("gemma4_text")(config)
        model = build_from_module(
            module,
            config,
            CausalLMTask(static_cache=True, max_seq_len=self.MAX_SEQ_LEN),
            execution_provider="qnn",
        )["model"]
        op_types = {n.op_type for n in model.graph}
        for forbidden in ("RotaryEmbedding", "TensorScatter", "Tile", "Range", "Attention"):
            assert forbidden not in op_types, f"{forbidden} should be lowered for qnn"
        assert "ScatterND" in op_types  # TensorScatter replacement
        assert "Expand" in op_types  # Tile (GQA repeat) replacement


class TestBuildGemma3nKvSharing:
    """Gemma 3n's trailing layers borrow K,V instead of projecting their own.

    Layers at or after ``num_hidden_layers - num_kv_shared_layers`` reuse the
    K,V of the last preceding layer of the *same* attention type, so they own
    no KV cache entry and carry no ``k_proj``/``v_proj``/``k_norm`` weights.
    """

    @staticmethod
    def _build(num_kv_shared_layers, layer_types):
        from mobius._configs import Gemma3nConfig

        config = Gemma3nConfig(
            num_hidden_layers=len(layer_types),
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=256,
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
            hidden_act="gelu_pytorch_tanh",
            rope_type="default",
            rope_theta=10_000.0,
            rope_local_base_freq=10_000.0,
            attn_qk_norm=True,
            layer_types=layer_types,
            sliding_window=8,
            altup_num_inputs=2,
            altup_active_idx=0,
            altup_correct_scale=True,
            laurel_rank=16,
            hidden_size_per_layer_input=32,
            vocab_size_per_layer_input=256,
            num_kv_shared_layers=num_kv_shared_layers,
            pad_token_id=0,
        )
        module = registry.get("gemma3n_text")(config)
        pkg = get_task(_default_task_for_model("gemma3n_text")).build(module, config)
        return module, pkg["model"], config

    def test_cache_io_excludes_shared_layers(self):
        """Only non-shared layers get past/present KV entries."""
        module, model, _config = self._build(2, ["full_attention"] * 4)

        assert module.kv_cache_layer_count() == 2
        input_names = {i.name for i in model.graph.inputs}
        output_names = {o.name for o in model.graph.outputs}
        for i in range(2):
            assert f"past_key_values.{i}.key" in input_names
            assert f"past_key_values.{i}.value" in input_names
            assert f"present.{i}.key" in output_names
            assert f"present.{i}.value" in output_names
        for i in (2, 3):
            assert f"past_key_values.{i}.key" not in input_names
            assert f"present.{i}.key" not in output_names

    def test_shared_layers_have_no_kv_weights(self):
        """KV-shared layers must not request k_proj/v_proj/k_norm initializers.

        The checkpoint ships these tensors for every layer, but HF only builds
        them for the non-shared layers — emitting them here would create
        initializers with no consumer.
        """
        _module, model, _config = self._build(2, ["full_attention"] * 4)

        names = set(model.graph.initializers)
        for i in (0, 1):
            assert f"model.layers.{i}.self_attn.k_proj.weight" in names
            assert f"model.layers.{i}.self_attn.v_proj.weight" in names
        for i in (2, 3):
            assert f"model.layers.{i}.self_attn.k_proj.weight" not in names
            assert f"model.layers.{i}.self_attn.v_proj.weight" not in names
            assert f"model.layers.{i}.self_attn.k_norm.weight" not in names

    def test_source_layer_matches_attention_type(self):
        """Sliding and full layers borrow from different source layers.

        HF indexes the *pre-cutoff* slice of ``layer_types`` for the matching
        type, so a shared sliding layer never borrows a full layer's K,V.
        """
        layer_types = [
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ]
        module, _model, _config = self._build(2, layer_types)

        attns = [layer.self_attn for layer in module.model.layers]
        assert [a.is_kv_shared_layer for a in attns] == [False, False, False, True, True]
        # Last non-shared layer of each type publishes its K,V for reuse.
        assert [a.provides_shared_kv for a in attns] == [False, True, True, False, False]
        # Shared sliding layer 3 -> layer 1; shared full layer 4 -> layer 2.
        assert attns[3].kv_shared_layer_index == 1
        assert attns[4].kv_shared_layer_index == 2

    def test_drops_shared_layer_weights_from_state_dict(self):
        """preprocess_weights discards the K/V tensors HF never constructs."""
        import torch

        module, _model, config = self._build(2, ["full_attention"] * 4)
        state_dict = {
            f"model.layers.{i}.self_attn.{name}.weight": torch.zeros(1)
            for i in range(config.num_hidden_layers)
            for name in ("q_proj", "k_proj", "v_proj", "k_norm")
        }

        result = module.preprocess_weights(state_dict)

        for i in range(config.num_hidden_layers):
            assert f"model.layers.{i}.self_attn.q_proj.weight" in result
        for i in (0, 1):
            assert f"model.layers.{i}.self_attn.k_proj.weight" in result
        for i in (2, 3):
            assert f"model.layers.{i}.self_attn.k_proj.weight" not in result
            assert f"model.layers.{i}.self_attn.v_proj.weight" not in result
            assert f"model.layers.{i}.self_attn.k_norm.weight" not in result

    def test_rejects_sharing_every_layer(self):
        """A layer cannot borrow K,V when no earlier layer computes any."""
        with pytest.raises(ValueError, match="num_kv_shared_layers"):
            self._build(4, ["full_attention"] * 4)


class TestBuildGemma4StaticCacheGraph:
    """Verify Gemma4TextCausalLMTask(static_cache=True) builds a valid graph."""

    MAX_SEQ_LEN = 128

    @staticmethod
    def _gemma4_config(**overrides):
        from mobius._configs import Gemma4Config

        defaults = dict(
            num_hidden_layers=6,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            global_head_dim=32,
            vocab_size=256,
            rms_norm_eps=1e-6,
            hidden_act="gelu_pytorch_tanh",
            # Mixed: 5 sliding + 1 full (Gemma4-like hybrid pattern)
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
            sliding_window=64,
            rope_theta=10000.0,
            global_rope_theta=1000000.0,
            partial_rotary_factor=0.5,
            max_position_embeddings=256,
            hidden_size_per_layer_input=0,
            num_kv_shared_layers=0,
        )
        defaults.update(overrides)
        return Gemma4Config(**defaults)

    def _build(self, **config_overrides):
        from mobius.models.gemma4 import Gemma4CausalLMModel
        from mobius.tasks._gemma4 import Gemma4TextCausalLMTask

        config = self._gemma4_config(**config_overrides)
        module = Gemma4CausalLMModel(config)
        task = Gemma4TextCausalLMTask(static_cache=True, max_seq_len=self.MAX_SEQ_LEN)
        pkg = task.build(module, config)
        return pkg["model"], config

    def test_gemma4_static_cache_builds(self):
        """Build Gemma4 with static cache and verify basic graph structure."""
        model, _config = self._build()

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "input_ids" in input_names
        assert "position_ids" in input_names
        # Hybrid mode: attention_mask for sliding GQA + write_indices for static
        assert "attention_mask" in input_names
        assert "write_indices" in input_names
        assert "nonpad_kv_seqlen" in input_names

    def test_gemma4_static_cache_hybrid_inputs(self):
        """Verify full-attention gets static cache, sliding gets dynamic."""
        model, _config = self._build()

        input_map = {inp.name: inp for inp in model.graph.inputs}

        # Layer 0-4: sliding → dynamic cache (past_key_values.N.key)
        assert "past_key_values.0.key" in input_map

        # Layer 5: full_attention → static cache (key_cache.5)
        assert "key_cache.5" in input_map
        kv_hidden_full = _config.num_key_value_heads * _config.global_head_dim
        k5 = input_map["key_cache.5"]
        assert k5.shape[2] == kv_hidden_full

    def test_gemma4_static_cache_has_tensorscatter(self):
        """Verify TensorScatter for full-attention layers in hybrid mode."""
        model, _config = self._build()

        op_counts = {}
        for n in model.graph:
            op_counts[n.op_type] = op_counts.get(n.op_type, 0) + 1

        layer_types = _config.layer_types or []
        num_full = sum(1 for lt in layer_types if lt == "full_attention")

        # TensorScatter: 2 per full-attention layer (key + value)
        assert op_counts.get("TensorScatter", 0) == 2 * num_full
        # Sliding layers use either GQA (CUDA EP) or Attention (default EP)
        # In unit tests without EP context, all use standard Attention.
        total_attn = op_counts.get("Attention", 0) + op_counts.get("GroupQueryAttention", 0)
        assert total_attn == _config.num_hidden_layers

    def test_gemma4_static_cache_kv_shared(self):
        """Verify KV-shared layers are excluded from cache I/O."""
        model, config = self._build(
            num_hidden_layers=8,
            num_kv_shared_layers=2,
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
        )

        input_names = {inp.name for inp in model.graph.inputs}
        num_kv_layers = config.num_hidden_layers - config.num_kv_shared_layers

        # Non-shared layers 0-5 should have cache entries (type depends on layer)
        for i in range(num_kv_layers):
            lt = config.layer_types[i]
            if lt == "full_attention":
                assert f"key_cache.{i}" in input_names
            else:
                assert f"past_key_values.{i}.key" in input_names

        # Shared layers (6, 7) should NOT have any cache entries
        assert f"key_cache.{num_kv_layers}" not in input_names
        assert f"past_key_values.{num_kv_layers}.key" not in input_names

    def test_gemma4_static_cache_input_ordering(self):
        """Verify write_indices/nonpad_kv_seqlen come after cache inputs."""
        model, _config = self._build()

        input_names = [inp.name for inp in model.graph.inputs]
        # write_indices and nonpad_kv_seqlen should come after all cache inputs
        last_cache_idx = max(i for i, n in enumerate(input_names) if "cache" in n)
        write_idx = input_names.index("write_indices")
        nonpad_idx = input_names.index("nonpad_kv_seqlen")
        assert write_idx > last_cache_idx, "write_indices should come after all cache inputs"
        assert nonpad_idx > last_cache_idx, (
            "nonpad_kv_seqlen should come after all cache inputs"
        )


# === Parametrized Vision-Language configs (imported from _test_configs) ===
_VL_MODEL_PARAMS = _make_params(VL_CONFIGS)

# VL models that produce a single "model" key instead of 3-model split
_VL_SINGLE_MODEL_TASKS = {"qwen3-vl-vision-language"}
_VL_TWO_MODEL_TASKS = {"vision-encoder-decoder"}


@pytest.mark.parametrize("model_type,config_overrides", _VL_MODEL_PARAMS)
class TestBuildVLGraph:
    """Verify vision-language models build valid multi-model ONNX packages."""

    def test_package_builds(self, model_type: str, config_overrides: dict):
        """Build a VL model and verify it produces the expected sub-models."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)

        if task_name in _VL_SINGLE_MODEL_TASKS:
            assert "model" in pkg, f"{model_type} should produce 'model'"
            model = pkg["model"]
            assert model.graph is not None
            output_names = {o.name for o in model.graph.outputs}
            assert "logits" in output_names
        elif task_name in _VL_TWO_MODEL_TASKS:
            assert set(pkg) == {"decoder", "vision_encoder"}
            decoder = pkg["decoder"]
            assert "encoder_hidden_states" in {i.name for i in decoder.graph.inputs}
            assert "logits" in {o.name for o in decoder.graph.outputs}
            vision = pkg["vision_encoder"]
            pixel_values = next(i for i in vision.graph.inputs if i.name == "pixel_values")
            assert pixel_values.dtype == ir.DataType.FLOAT
        else:
            assert "decoder" in pkg, f"{model_type} should produce 'decoder'"
            assert "vision_encoder" in pkg, f"{model_type} should produce 'vision_encoder'"
            assert "embedding" in pkg, f"{model_type} should produce 'embedding'"

            decoder = pkg["decoder"]
            assert "inputs_embeds" in {i.name for i in decoder.graph.inputs}
            assert "logits" in {o.name for o in decoder.graph.outputs}

            vision = pkg["vision_encoder"]
            pixel_values = next(i for i in vision.graph.inputs if i.name == "pixel_values")
            assert pixel_values.dtype == ir.DataType.FLOAT

    def test_has_initializers(self, model_type: str, config_overrides: dict):
        """Verify all sub-models have non-empty initializers."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)

        for name, model in pkg.items():
            init_names = list(model.graph.initializers)
            assert len(init_names) > 0, f"{model_type}/{name} should have initializers"

    def test_onnx_checker_passes(self, model_type: str, config_overrides: dict):
        """Run ONNX CheckerPass on all sub-models."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = get_task(_default_task_for_model(model_type))
        pkg = task.build(module, config)
        _run_onnx_checker(pkg, model_type)

    def test_outputs_have_shapes_and_dtypes(self, model_type: str, config_overrides: dict):
        """Verify shape inference populates all output shapes and dtypes."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = get_task(_default_task_for_model(model_type))
        pkg = task.build(module, config)
        _assert_outputs_have_shapes_and_dtypes(pkg, model_type)


# === Parametrized Speech / TTS / Codec configs ===
_SPEECH_MODEL_PARAMS = _make_params(SPEECH_CONFIGS)

# Expected sub-model keys per speech task type
_SPEECH_TASK_KEYS: dict[str, set[str]] = {
    "speech-to-text": {"encoder", "decoder"},
    "speech-language": {"audio_encoder", "embedding", "decoder"},
    "codec": {"decoder", "encoder"},
    "audio-feature-extraction": {"model"},
}


@pytest.mark.parametrize("model_type,config_overrides", _SPEECH_MODEL_PARAMS)
class TestBuildSpeechGraph:
    """Verify speech/TTS/codec models build valid multi-model packages."""

    def test_package_builds(self, model_type: str, config_overrides: dict):
        """Build a speech model and verify expected sub-models."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)

        expected = _SPEECH_TASK_KEYS.get(task_name, set())
        for key in expected:
            assert key in pkg, f"{model_type} ({task_name}) should produce '{key}'"

        # Every sub-model should have a valid graph
        for name, model in pkg.items():
            assert model.graph is not None, f"{model_type}/{name} graph is None"
            assert len(model.graph.inputs) > 0, f"{model_type}/{name} has no inputs"
            assert len(model.graph.outputs) > 0, f"{model_type}/{name} has no outputs"

    def test_has_initializers(self, model_type: str, config_overrides: dict):
        """Verify all sub-models have non-empty initializers."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)

        for name, model in pkg.items():
            init_names = list(model.graph.initializers)
            assert len(init_names) > 0, f"{model_type}/{name} should have initializers"

    def test_onnx_checker_passes(self, model_type: str, config_overrides: dict):
        """Run ONNX CheckerPass on all sub-models."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = get_task(_default_task_for_model(model_type))
        pkg = task.build(module, config)
        _run_onnx_checker(pkg, model_type)

    def test_outputs_have_shapes_and_dtypes(self, model_type: str, config_overrides: dict):
        """Verify shape inference populates all output shapes and dtypes."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = get_task(_default_task_for_model(model_type))
        pkg = task.build(module, config)
        _assert_outputs_have_shapes_and_dtypes(pkg, model_type)


class TestGQASlidingWindow:
    """Wire ``config.sliding_window`` into GQA's ``local_window_size``.

    On the direct GQA path (``TextModel.forward``), GQA
    ``local_window_size=W`` masks each query to the most recent ``W`` keys
    (positions ``[i-W+1, i]``), matching HuggingFace ``sliding_window=W``.
    Because the global GQAContext is shared by every layer, the window is only
    emitted for models with a *uniform* sliding window across all layers.
    """

    @staticmethod
    def _build_gqa_model(**overrides):
        from mobius.models.base import CausalLMModel

        overrides.setdefault("num_hidden_layers", 2)
        config = ArchitectureConfig(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=256,
            max_position_embeddings=128,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            rope_type="default",
            rope_theta=10000.0,
            pad_token_id=0,
            dtype=ir.DataType.FLOAT16,
            **overrides,
        )
        module = CausalLMModel(config)
        # execution_provider="cuda" + fp16 activates the direct GQA path.
        return build_from_module(module, config, execution_provider="cuda")["model"]

    @classmethod
    def _build_gqa_decoder(cls, **overrides):
        model = cls._build_gqa_model(**overrides)
        gqa_nodes = [n for n in model.graph if n.op_type == "GroupQueryAttention"]
        assert gqa_nodes, "expected GroupQueryAttention nodes on the cuda/fp16 path"
        return gqa_nodes

    @staticmethod
    def _fill_fp16_weights(model, seed):
        """Deterministically fill uninitialised params, honouring fp16 dtype."""
        rng = np.random.default_rng(seed)
        npdt = {ir.DataType.FLOAT16: np.float16, ir.DataType.FLOAT: np.float32}
        for init in model.graph.initializers.values():
            if init.const_value is None:
                shape = [int(d) for d in init.shape]
                arr = (rng.standard_normal(shape) * 0.1).astype(
                    npdt.get(init.dtype, np.float32)
                )
                init.const_value = ir.Tensor(arr)

    def test_uniform_sliding_window_sets_local_window_size(self):
        """A uniform sliding window is forwarded to every GQA node verbatim."""
        gqa_nodes = self._build_gqa_decoder(sliding_window=3000)
        for node in gqa_nodes:
            assert node.attributes["local_window_size"].value == 3000

    def test_no_sliding_window_omits_local_window_size(self):
        """Full-attention models must not carry a local_window_size attribute."""
        gqa_nodes = self._build_gqa_decoder(sliding_window=None)
        for node in gqa_nodes:
            assert "local_window_size" not in node.attributes

    def test_mixed_layer_types_omits_local_window_size(self):
        """A per-layer schedule cannot be expressed by one global window."""
        gqa_nodes = self._build_gqa_decoder(
            sliding_window=8,
            layer_types=["sliding_attention", "full_attention"],
        )
        for node in gqa_nodes:
            assert "local_window_size" not in node.attributes

    def test_all_sliding_layer_types_sets_local_window_size(self):
        """An explicit all-sliding schedule is uniform, so window is emitted."""
        gqa_nodes = self._build_gqa_decoder(
            sliding_window=8,
            layer_types=["sliding_attention", "sliding_attention"],
        )
        for node in gqa_nodes:
            assert node.attributes["local_window_size"].value == 8

    def test_window_confines_receptive_field(self):
        """The sliding window must actually bound each query's receptive field.

        Property test (no golden, weight-agnostic): with window ``W`` over
        ``L`` layers, the last position can only be influenced by inputs within
        ``L*(W-1)`` steps. Perturbing an out-of-window token must leave the
        windowed model's last-position logits unchanged, while the same
        perturbation *does* change the full-attention twin's logits — proving
        the window is genuinely applied (and exercising seq > window, unlike
        the short-sequence Moshi parity golden). Runs the fp16 GQA op on CPU.
        """
        from mobius._testing.ort_inference import OnnxModelSession

        window, seq, layers = 4, 24, 2
        # Pos 0 is well outside the last position's receptive field
        # (layers*(window-1) = 6 << seq-1 = 23).
        windowed = self._build_gqa_model(sliding_window=window, num_hidden_layers=layers)
        full = self._build_gqa_model(sliding_window=None, num_hidden_layers=layers)
        # Same seed + identical structure (only the GQA attr differs) => same weights.
        self._fill_fp16_weights(windowed, seed=1234)
        self._fill_fp16_weights(full, seed=1234)

        def feeds(first_token):
            ids = np.arange(1, seq + 1, dtype=np.int64).reshape(1, seq)
            ids[0, 0] = first_token
            f = {"input_ids": ids, "attention_mask": np.ones((1, seq), np.int64)}
            for i in range(layers):
                f[f"past_key_values.{i}.key"] = np.zeros((1, 2, 0, 16), np.float16)
                f[f"past_key_values.{i}.value"] = np.zeros((1, 2, 0, 16), np.float16)
            return f

        def last_logits(model, first_token):
            sess = OnnxModelSession(model)
            out = sess.run(feeds(first_token))
            return out["logits"][0, -1].astype(np.float32)

        # Windowed: perturbing the out-of-window first token leaves the last
        # position's logits unchanged.
        w_a = last_logits(windowed, first_token=5)
        w_b = last_logits(windowed, first_token=200)
        np.testing.assert_allclose(w_a, w_b, atol=1e-3)

        # Full attention: the same perturbation DOES reach the last position.
        f_a = last_logits(full, first_token=5)
        f_b = last_logits(full, first_token=200)
        assert np.abs(f_a - f_b).max() > 1e-2, (
            "full-attention twin should be sensitive to the first token"
        )

    def test_empty_layer_types_omits_local_window_size(self):
        """An empty ``layer_types`` is not a valid uniform schedule.

        ``all(... for t in [])`` is vacuously True, so an unhardened guard
        would wrongly treat ``[]`` as uniform-sliding. The length check against
        ``num_hidden_layers`` rejects it, leaving the window disabled.
        """
        gqa_nodes = self._build_gqa_decoder(
            sliding_window=8,
            layer_types=[],
            num_hidden_layers=2,
        )
        for node in gqa_nodes:
            assert "local_window_size" not in node.attributes

    def test_non_gqa_path_warns_on_uniform_window(self, caplog):
        """Non-GQA (static-cache) export of a uniform-sliding model warns.

        That path cannot represent the window, so the warning flags the
        divergence from HuggingFace for sequences longer than the window.
        """
        import logging

        from mobius.models.base import CausalLMModel

        config = ArchitectureConfig(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=256,
            max_position_embeddings=128,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            rope_type="default",
            rope_theta=10000.0,
            pad_token_id=0,
            num_hidden_layers=2,
            sliding_window=8,
            dtype=ir.DataType.FLOAT,
        )
        module = CausalLMModel(config)
        # Static cache feeds attention_mask=None, taking the non-GQA else
        # branch that cannot express the window.
        from mobius.tasks import CausalLMTask

        task = CausalLMTask(static_cache=True, max_seq_len=128)
        with caplog.at_level(logging.WARNING, logger="mobius.models.base"):
            build_from_module(module, config, task=task, execution_provider="cpu")
        assert "sliding window" in caplog.text

    def test_partial_layer_types_omits_local_window_size(self):
        """A ``layer_types`` shorter than the layer count is not uniform."""
        gqa_nodes = self._build_gqa_decoder(
            sliding_window=8,
            layer_types=["sliding_attention"],
            num_hidden_layers=2,
        )
        for node in gqa_nodes:
            assert "local_window_size" not in node.attributes


class TestResolveSlidingWindow:
    """``from_transformers`` must honor HF's ``use_sliding_window`` gate.

    Qwen2/Qwen3 keep a non-null ``sliding_window`` in the config even when the
    window is disabled and signal activation via ``use_sliding_window``. A raw
    ``config.json`` fallback bypasses HF's ``__post_init__`` (which would null
    the field), so the gate is re-applied in ``_resolve_sliding_window``.
    """

    @staticmethod
    def _cfg(**attrs):
        from types import SimpleNamespace

        defaults = dict(
            model_type="qwen2",
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
            vocab_size=256,
            max_position_embeddings=128,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
        )
        defaults.update(attrs)
        return SimpleNamespace(**defaults)

    def test_disabled_window_is_nulled(self):
        """``use_sliding_window=False`` drops a non-null ``sliding_window``."""
        cfg = ArchitectureConfig.from_transformers(
            self._cfg(sliding_window=4096, use_sliding_window=False), "qwen2"
        )
        assert cfg.sliding_window is None

    def test_enabled_window_is_kept(self):
        """``use_sliding_window=True`` keeps the window."""
        cfg = ArchitectureConfig.from_transformers(
            self._cfg(sliding_window=4096, use_sliding_window=True), "qwen2"
        )
        assert cfg.sliding_window == 4096

    def test_window_without_flag_is_kept(self):
        """Models without ``use_sliding_window`` (e.g. Mistral) are unaffected."""
        cfg = ArchitectureConfig.from_transformers(
            self._cfg(model_type="mistral", sliding_window=4096), "mistral"
        )
        assert cfg.sliding_window == 4096


class TestLongRopeAliasExtraction:
    """``rope_type`` alias handling for Phi LongRoPE.

    Phi-3/Phi-3.5 checkpoints label LongRoPE as ``"su"`` (short/long-factor
    scaled rotary embeddings); newer HuggingFace configs spell the identical
    algorithm ``"longrope"``. ``_extract_rope_config`` must canonicalize the
    legacy ``"su"`` alias to ``"longrope"`` so both configs resolve to the
    same ``LongRope`` code path.
    """

    _SHORT_FACTOR = [1.0] * (TINY_HEAD_DIM // 2)
    _LONG_FACTOR = [2.0] * (TINY_HEAD_DIM // 2)

    @staticmethod
    def _cfg(rope_scaling, **attrs):
        from types import SimpleNamespace

        defaults = dict(
            model_type="phi3",
            hidden_size=TINY_HIDDEN,
            intermediate_size=TINY_INTERMEDIATE,
            num_attention_heads=TINY_HEADS,
            num_key_value_heads=TINY_KV_HEADS,
            head_dim=TINY_HEAD_DIM,
            num_hidden_layers=TINY_LAYERS,
            vocab_size=TINY_VOCAB,
            max_position_embeddings=1024,
            original_max_position_embeddings=128,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            rope_scaling=rope_scaling,
        )
        defaults.update(attrs)
        return SimpleNamespace(**defaults)

    def _scaling(self, rope_type_key, rope_type_value):
        return {
            rope_type_key: rope_type_value,
            "short_factor": self._SHORT_FACTOR,
            "long_factor": self._LONG_FACTOR,
        }

    def test_su_and_longrope_produce_identical_rope_config(self):
        """``type="su"`` and ``rope_type="longrope"`` extract identically."""
        from mobius._configs._base import _extract_rope_config

        su_config = _extract_rope_config(self._cfg(self._scaling("type", "su")))
        longrope_config = _extract_rope_config(
            self._cfg(self._scaling("rope_type", "longrope"))
        )

        assert su_config is not None
        assert su_config.rope_type == "longrope"
        assert longrope_config.rope_type == "longrope"
        assert su_config.original_max_position_embeddings == 128
        assert su_config.rope_type == longrope_config.rope_type
        assert (
            su_config.rope_scaling["short_factor"]
            == longrope_config.rope_scaling["short_factor"]
        )
        assert (
            su_config.rope_scaling["long_factor"]
            == longrope_config.rope_scaling["long_factor"]
        )
        assert (
            su_config.original_max_position_embeddings
            == longrope_config.original_max_position_embeddings
        )

    def test_su_alias_dispatches_to_longrope_module(self):
        """A ``"su"`` config resolves to the ``LongRope`` runtime module."""
        from mobius.components._rotary_embedding import LongRope, initialize_rope

        config = ArchitectureConfig.from_transformers(self._cfg(self._scaling("type", "su")))
        assert config.rope_type == "longrope"
        rope = initialize_rope(config)
        assert isinstance(rope, LongRope)

    def test_su_graph_builds_end_to_end(self):
        """A phi3 ``"su"`` config builds a valid ONNX graph without weights."""
        config = ArchitectureConfig.from_transformers(self._cfg(self._scaling("type", "su")))
        module = registry.get("phi3")(config)
        task = get_task(_default_task_for_model("phi3"))
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "input_ids" in input_names
        assert "position_ids" in input_names
        output_names = {out.name for out in model.graph.outputs}
        assert "logits" in output_names

    def test_missing_original_max_position_embeddings_still_maps_to_longrope(self):
        """A ``"su"`` config without ``original_max_position_embeddings``.

        The alias must still resolve to ``longrope`` and ``LongRope`` falls
        back to ``max_position_embeddings`` for the short cache length rather
        than crashing.
        """
        from mobius._configs._base import _extract_rope_config
        from mobius.components._rotary_embedding import LongRope, initialize_rope

        config_source = self._cfg(
            self._scaling("type", "su"), original_max_position_embeddings=None
        )
        rope_config = _extract_rope_config(config_source)
        assert rope_config.rope_type == "longrope"
        assert rope_config.original_max_position_embeddings is None

        arch_config = ArchitectureConfig.from_transformers(config_source)
        rope = initialize_rope(arch_config)
        assert isinstance(rope, LongRope)

    def test_factor_length_mismatch_is_rejected(self):
        """Short/long factor arrays must match the rotary dimension.

        A factor list whose length does not equal ``head_dim / 2`` cannot be
        broadcast against the inverse-frequency vector, so ``LongRope``
        construction raises rather than silently producing a wrong cache.
        """
        from mobius.components._rotary_embedding import initialize_rope

        bad_scaling = {
            "type": "su",
            "short_factor": [1.0] * (TINY_HEAD_DIM // 2 + 1),
            "long_factor": [2.0] * (TINY_HEAD_DIM // 2 + 1),
        }
        config = ArchitectureConfig.from_transformers(self._cfg(bad_scaling))
        assert config.rope_type == "longrope"
        with pytest.raises(ValueError, match="broadcast"):
            initialize_rope(config)

    def test_non_alias_rope_types_are_unchanged(self):
        """Canonicalization only rewrites known aliases, not other types."""
        from mobius._configs._base import _canonical_rope_type

        assert _canonical_rope_type("su") == "longrope"
        assert _canonical_rope_type("longrope") == "longrope"
        assert _canonical_rope_type("yarn") == "yarn"
        assert _canonical_rope_type("default") == "default"
        assert _canonical_rope_type(None) is None


class TestBuildGraphSortformer:
    """Verify Sortformer diarization builds with DiarizationTask."""

    def _sortformer_config(self):
        from mobius.models.sortformer import SortformerConfig

        # Tiny config: reduced widths/layers, structure identical to the real
        # nvidia/diar_streaming_sortformer_4spk model.
        return SortformerConfig(
            feat_in=32,
            fc_d_model=64,
            fc_num_layers=2,
            fc_num_heads=4,
            fc_ff_expansion=4,
            fc_conv_kernel=9,
            fc_subsampling_conv_channels=16,
            fc_subsampling_factor=8,
            tf_d_model=32,
            tf_num_layers=2,
            tf_num_heads=4,
            tf_inner_size=64,
            num_spks=4,
        )

    def test_package_builds(self):
        """Build Sortformer and verify a single 'model' component."""
        from mobius.models.sortformer import SortformerDiarizationModel
        from mobius.tasks import DiarizationTask

        config = self._sortformer_config()
        module = SortformerDiarizationModel(config)
        pkg = build_from_module(module, config, task=DiarizationTask())

        assert "model" in pkg

    def test_model_io(self):
        """Verify diarization input/output names and shapes."""
        from mobius.models.sortformer import SortformerDiarizationModel
        from mobius.tasks import DiarizationTask

        config = self._sortformer_config()
        module = SortformerDiarizationModel(config)
        pkg = build_from_module(module, config, task=DiarizationTask())
        model = pkg["model"]

        input_names = {inp.name for inp in model.graph.inputs}
        output_names = {out.name for out in model.graph.outputs}
        assert "input_features" in input_names
        assert "speaker_probs" in output_names

        # input_features: [batch, feat_in, time]
        feat_dim = model.graph.inputs[0].shape[1]
        assert feat_dim == config.feat_in
        # speaker_probs last dim == num_spks
        spk_dim = model.graph.outputs[0].shape[2]
        assert spk_dim == config.num_spks

    def test_has_initializers(self):
        """Verify encoder / transformer / head initializers are present."""
        from mobius.models.sortformer import SortformerDiarizationModel
        from mobius.tasks import DiarizationTask

        config = self._sortformer_config()
        module = SortformerDiarizationModel(config)
        pkg = build_from_module(module, config, task=DiarizationTask())
        init_names = list(pkg["model"].graph.initializers)

        assert any(n.startswith("encoder.") for n in init_names)
        assert any(n.startswith("transformer_encoder.") for n in init_names)
        assert any(n.startswith("sortformer_modules.") for n in init_names)

    def test_task_registry_lookup(self):
        """Verify the 'diarization' task resolves to DiarizationTask."""
        from mobius.tasks import DiarizationTask, get_task

        assert isinstance(get_task("diarization"), DiarizationTask)

    def test_runs_with_random_weights(self):
        """Fill random weights and run a forward pass through ORT."""
        import os
        import tempfile

        import onnxruntime as ort

        from mobius.models.sortformer import SortformerDiarizationModel
        from mobius.tasks import DiarizationTask

        config = self._sortformer_config()
        module = SortformerDiarizationModel(config)
        pkg = build_from_module(module, config, task=DiarizationTask())
        model = pkg["model"]

        # Fill each empty initializer with small random values.  Batch-norm
        # running variance must stay positive to avoid NaNs.
        for init in model.graph.initializers.values():
            if init.const_value is not None:
                continue
            shape = [d if isinstance(d, int) else 1 for d in init.shape]
            if "running_var" in init.name:
                arr = np.ones(shape, dtype=np.float32)
            elif "running_mean" in init.name:
                arr = np.zeros(shape, dtype=np.float32)
            else:
                arr = (np.random.randn(*shape) * 0.02).astype(np.float32)
            init.const_value = ir.tensor(arr, name=init.name)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.onnx")
            ir.save(model, path, external_data="model.onnx.data")
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            n_time = 40  # multiple of subsampling factor (8) -> 5 output frames
            feats = np.random.randn(1, config.feat_in, n_time).astype(np.float32)
            out = sess.run(None, {sess.get_inputs()[0].name: feats})[0]

        assert out.shape == (1, n_time // config.fc_subsampling_factor, config.num_spks)
        # Sigmoid output must lie in [0, 1].
        assert out.min() >= 0.0 and out.max() <= 1.0
