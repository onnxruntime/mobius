# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Gemma3n activation sparsity and multimodal weight renaming."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch

from mobius._configs import (
    Gemma3nAudioConfig,
    Gemma3nConfig,
    Gemma3nMultiModalConfig,
    VisionConfig,
)
from mobius._constants import OPSET_VERSION
from mobius.models.gemma3n import Gemma3nMLP, Gemma3nMultiModalModel


def _tiny_gemma3n_config(**overrides) -> Gemma3nConfig:
    """Create a minimal Gemma3nConfig for MLP-level tests."""
    defaults = dict(
        model_type="gemma3n_text",
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        head_dim=16,
        hidden_act="gelu_pytorch_tanh",
        layer_types=["full_attention", "sliding_attention"],
        attn_qk_norm=True,
        altup_num_inputs=2,
        laurel_rank=16,
        hidden_size_per_layer_input=32,
        vocab_size_per_layer_input=256,
    )
    defaults.update(overrides)
    return Gemma3nConfig(**defaults)


def _hf_gaussian_topk(inputs: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Verbatim port of HF ``Gemma3nTextMLP._gaussian_topk``."""
    target = torch.tensor(sparsity, dtype=torch.float32, device=inputs.device)
    std_multiplier = torch.distributions.normal.Normal(0, 1).icdf(target)
    std_multiplier = std_multiplier.type(inputs.dtype)
    inputs_mean = torch.mean(inputs, dim=-1, keepdim=True)
    inputs_std = torch.std(inputs, dim=-1, keepdim=True, unbiased=False)
    cutoff_x = inputs_mean + inputs_std * std_multiplier
    return torch.nn.functional.relu(inputs - cutoff_x)


def _run_gaussian_topk(mlp: Gemma3nMLP, x: np.ndarray) -> np.ndarray:
    """Build a single-op ONNX graph around ``mlp._gaussian_topk`` and run it."""
    from onnxscript import GraphBuilder

    x_input = ir.Value(
        name="x",
        shape=ir.Shape(list(x.shape)),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    graph = ir.Graph(
        inputs=[x_input],
        outputs=[],
        nodes=[],
        name="test_gaussian_topk",
        opset_imports={"": OPSET_VERSION},
    )
    gb = GraphBuilder(graph)
    result = mlp._gaussian_topk(gb.op, x_input)
    result.name = "output"
    graph.outputs.append(result)

    # Serialize in-memory (avoids Windows PermissionError from concurrent
    # tempfile access), matching the other component-level ORT tests.
    proto = ir.serde.serialize_model(ir.Model(graph, ir_version=11))
    sess = ort.InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, {"x": x})[0]


class TestGemma3nActivationSparsity:
    """Gemma 3n sparsifies the gate projection on its early layers.

    E4B applies 0.95 sparsity to layers 0-9 and none to layers 10-34, so the
    per-layer pattern must select between the sparse and plain MLP paths.
    """

    @pytest.mark.parametrize("sparsity", [0.95, 0.5, 0.1])
    def test_std_multiplier_matches_torch_icdf(self, sparsity):
        """The folded Phi^-1(sparsity) constant matches HF's icdf call."""
        config = _tiny_gemma3n_config(activation_sparsity_pattern=[sparsity, 0.0])
        mlp = Gemma3nMLP(config, layer_idx=0)

        expected = (
            torch.distributions.normal.Normal(0, 1)
            .icdf(torch.tensor(sparsity, dtype=torch.float32))
            .item()
        )
        assert mlp._std_multiplier == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize("sparsity", [0.95, 0.5])
    def test_gaussian_topk_matches_hf(self, sparsity):
        """The ONNX cutoff graph matches HF's ``_gaussian_topk`` numerically."""
        config = _tiny_gemma3n_config(activation_sparsity_pattern=[sparsity, 0.0])
        mlp = Gemma3nMLP(config, layer_idx=0)

        rng = np.random.default_rng(0)
        x = rng.standard_normal((2, 3, 128)).astype(np.float32)

        onnx_out = _run_gaussian_topk(mlp, x)
        hf_out = _hf_gaussian_topk(torch.from_numpy(x), sparsity).numpy()

        np.testing.assert_allclose(onnx_out, hf_out, atol=1e-5, rtol=1e-5)

    def test_gaussian_topk_zeroes_expected_fraction(self):
        """0.95 sparsity keeps roughly 5% of a Gaussian row's activations."""
        config = _tiny_gemma3n_config(activation_sparsity_pattern=[0.95, 0.0])
        mlp = Gemma3nMLP(config, layer_idx=0)

        rng = np.random.default_rng(0)
        x = rng.standard_normal((8, 1024)).astype(np.float32)

        kept = (_run_gaussian_topk(mlp, x) > 0).mean()
        assert kept == pytest.approx(0.05, abs=0.02)

    def test_pattern_selects_per_layer_sparsity(self):
        """Each layer reads its own entry from the pattern."""
        config = _tiny_gemma3n_config(activation_sparsity_pattern=[0.95, 0.0])

        assert Gemma3nMLP(config, layer_idx=0).activation_sparsity == pytest.approx(0.95)
        assert Gemma3nMLP(config, layer_idx=1).activation_sparsity == pytest.approx(0.0)

    def test_no_pattern_disables_sparsity(self):
        """Without a pattern the MLP keeps the plain gated path."""
        mlp = Gemma3nMLP(_tiny_gemma3n_config(), layer_idx=0)

        assert mlp.activation_sparsity == pytest.approx(0.0)
        assert mlp._std_multiplier == pytest.approx(0.0)

    def test_dense_layer_emits_no_cutoff_ops(self):
        """A zero-sparsity layer must not pay for the mean/std subgraph."""
        from onnxscript import GraphBuilder

        config = _tiny_gemma3n_config(activation_sparsity_pattern=[0.95, 0.0])
        graph = ir.Graph(
            inputs=[],
            outputs=[],
            nodes=[],
            name="test_dense",
            opset_imports={"": OPSET_VERSION},
        )
        gb = GraphBuilder(graph)
        x = ir.Value(
            name="x",
            shape=ir.Shape([1, 2, config.hidden_size]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )
        graph.inputs.append(x)

        Gemma3nMLP(config, layer_idx=1).forward(gb.op, x)

        assert "Relu" not in {node.op_type for node in graph}

    @pytest.mark.parametrize("sparsity", [1.0, -0.1, 1.5])
    def test_rejects_out_of_range_sparsity(self, sparsity):
        """Sparsity outside [0, 1) has no finite Gaussian cutoff."""
        config = _tiny_gemma3n_config(activation_sparsity_pattern=[sparsity, 0.0])

        with pytest.raises(ValueError, match="activation_sparsity_pattern"):
            Gemma3nMLP(config, layer_idx=0)

    def test_rejects_short_pattern(self):
        """A pattern that does not cover every layer is a config error."""
        config = _tiny_gemma3n_config(activation_sparsity_pattern=[0.95])

        with pytest.raises(ValueError, match="must cover every layer"):
            Gemma3nMLP(config, layer_idx=1)


# ---------------------------------------------------------------------------
# Multimodal weight renaming
# ---------------------------------------------------------------------------
#: HF ships every Gemma 3n multimodal key under this prefix.
_HF = "model."


def _tiny_multimodal_config(**overrides) -> Gemma3nMultiModalConfig:
    """Create a minimal Gemma3nMultiModalConfig for renaming tests.

    Only ``preprocess_weights`` is exercised here, so the towers' sizes are
    irrelevant — but they must be *present*, since which sub-models exist
    decides where keys are routed.  Kept independent of the graph-level
    ``VL_CONFIGS`` entry, which has to satisfy MobileNet-V5's ``image_size``
    constraints that no rename depends on.
    """
    defaults = dict(
        model_type="gemma3n",
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        head_dim=16,
        hidden_act="gelu_pytorch_tanh",
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_type="default",
        rope_theta=10_000.0,
        rope_local_base_freq=10_000.0,
        layer_types=["full_attention", "sliding_attention"],
        attn_qk_norm=True,
        altup_num_inputs=2,
        laurel_rank=16,
        hidden_size_per_layer_input=32,
        vocab_size_per_layer_input=256,
        image_token_id=216,
        audio_token_id=217,
        vision_soft_tokens_per_image=256,
        audio_soft_tokens_per_image=8,
        vision=VisionConfig(
            hidden_size=32,
            image_size=256,
            rms_norm_eps=1e-6,
            vocab_offset=200,
            vocab_size=8,
            architecture="mobilenetv5_300m_enc",
            do_pooling=False,
        ),
        audio=Gemma3nAudioConfig(
            hidden_size=32,
            conf_num_attention_heads=4,
            conf_num_hidden_layers=1,
            conf_attention_chunk_size=4,
            conf_attention_context_left=5,
            conf_attention_context_right=0,
            conf_reduction_factor=2,
            input_feat_size=16,
            sscp_conv_channel_size=[8, 4],
            vocab_offset=208,
            vocab_size=8,
        ),
    )
    defaults.update(overrides)
    return Gemma3nMultiModalConfig(**defaults)


class TestGemma3nMultiModalPreprocessWeights:
    """``Gemma3nMultiModalModel.preprocess_weights`` HF -> ONNX renaming.

    The original gemma3n export was silently vision-blind because the
    causal-LM ``preprocess_weights`` *deleted* every ``vision_tower.`` and
    ``audio_tower.`` key.  These tests pin the routing table so no key can go
    missing again without failing here.
    """

    def test_language_model_routes_to_decoder(self):
        model = Gemma3nMultiModalModel(_tiny_multimodal_config())

        result = model.preprocess_weights(
            {
                f"{_HF}language_model.layers.0.mlp.gate_proj.weight": torch.zeros(1),
                f"{_HF}language_model.norm.weight": torch.zeros(1),
            }
        )

        assert "decoder.model.layers.0.mlp.gate_proj.weight" in result
        assert "decoder.model.norm.weight" in result

    def test_lm_head_skips_the_model_level(self):
        """``lm_head`` hangs off the decoder root, not ``decoder.model``."""
        model = Gemma3nMultiModalModel(_tiny_multimodal_config())

        result = model.preprocess_weights(
            {f"{_HF}language_model.lm_head.weight": torch.zeros(1)}
        )

        assert "decoder.lm_head.weight" in result
        assert "decoder.model.lm_head.weight" not in result

    def test_tied_checkpoint_synthesizes_lm_head(self):
        """E4B ships no ``lm_head``; it is tied to the token embedding."""
        embed = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        model = Gemma3nMultiModalModel(_tiny_multimodal_config(tie_word_embeddings=True))

        result = model.preprocess_weights(
            {f"{_HF}language_model.embed_tokens.weight": embed.clone()}
        )

        assert torch.equal(result["decoder.lm_head.weight"], embed)

    def test_untied_checkpoint_keeps_its_own_lm_head(self):
        """An explicit head must not be overwritten by the tying fallback."""
        head = torch.ones(3, 2)
        model = Gemma3nMultiModalModel(_tiny_multimodal_config(tie_word_embeddings=True))

        result = model.preprocess_weights(
            {
                f"{_HF}language_model.embed_tokens.weight": torch.zeros(3, 2),
                f"{_HF}language_model.lm_head.weight": head.clone(),
            }
        )

        assert torch.equal(result["decoder.lm_head.weight"], head)

    def test_no_lm_head_when_untied(self):
        """Without tying, a headless state dict stays headless.

        A load error downstream is better than silently reusing the embedding.
        """
        model = Gemma3nMultiModalModel(_tiny_multimodal_config(tie_word_embeddings=False))

        result = model.preprocess_weights(
            {f"{_HF}language_model.embed_tokens.weight": torch.zeros(3, 2)}
        )

        assert "decoder.lm_head.weight" not in result

    def test_embed_tokens_goes_to_both_decoder_and_embedding(self):
        model = Gemma3nMultiModalModel(_tiny_multimodal_config())

        result = model.preprocess_weights(
            {f"{_HF}language_model.embed_tokens.weight": torch.zeros(3, 2)}
        )

        assert "embedding.embed_tokens.weight" in result
        assert "decoder.model.embed_tokens.weight" in result

    @pytest.mark.parametrize(
        "suffix",
        [
            "embed_tokens_per_layer.weight",
            "per_layer_model_projection.weight",
            "per_layer_projection_norm.weight",
        ],
    )
    def test_per_layer_tables_go_only_to_embedding(self, suffix):
        """The 4.7 GB per-layer table must not be duplicated into the decoder."""
        model = Gemma3nMultiModalModel(_tiny_multimodal_config())

        result = model.preprocess_weights({f"{_HF}language_model.{suffix}": torch.zeros(1)})

        assert f"embedding.{suffix}" in result
        assert not any(key.startswith("decoder.") for key in result)

    def test_altup_level_is_stripped(self):
        """AltUp submodules register on the parent layer, without the level."""
        model = Gemma3nMultiModalModel(_tiny_multimodal_config())

        result = model.preprocess_weights(
            {
                f"{_HF}language_model.layers.0.altup.correction_coefs.weight": torch.zeros(1),
                f"{_HF}language_model.layers.0.altup.correct_output_scale": torch.zeros(1),
            }
        )

        assert "decoder.model.layers.0.correction_coefs.weight" in result
        assert "decoder.model.layers.0.correct_output_scale" in result
        assert not any(".altup." in key for key in result)

    def test_vision_tower_drops_the_timm_wrapper_level(self):
        model = Gemma3nMultiModalModel(_tiny_multimodal_config())

        result = model.preprocess_weights(
            {f"{_HF}vision_tower.timm_model.blocks.0.0.conv_exp.weight": torch.zeros(1)}
        )

        assert "vision_encoder.encoder.blocks.0.0.conv_exp.weight" in result
        assert not any("timm_model" in key for key in result)

    def test_audio_tower_routes_to_audio_encoder(self):
        model = Gemma3nMultiModalModel(_tiny_multimodal_config())

        result = model.preprocess_weights(
            {f"{_HF}audio_tower.conformer.0.attention.post.weight": torch.zeros(1)}
        )

        assert "audio_encoder.encoder.conformer.0.attention.post.weight" in result

    @pytest.mark.parametrize("modality", ["vision", "audio"])
    def test_embedder_weights_are_duplicated_into_two_components(self, modality):
        """Both graphs need every tensor of both embedders.

        Each embedder is used soft-path in its own tower and hard-path in the
        embedding graph.
        """
        component = "vision_encoder" if modality == "vision" else "audio_encoder"
        model = Gemma3nMultiModalModel(_tiny_multimodal_config())

        result = model.preprocess_weights(
            {
                f"{_HF}embed_{modality}.{name}": torch.zeros(1)
                for name in (
                    "embedding.weight",
                    "embedding_projection.weight",
                    "hard_embedding_norm.weight",
                    "soft_embedding_norm.weight",
                )
            }
        )

        for prefix in (f"{component}.embed_{modality}.", f"embedding.embed_{modality}."):
            assert f"{prefix}embedding.weight" in result, prefix
            assert f"{prefix}embedding_projection.weight" in result, prefix
            assert f"{prefix}hard_embedding_norm.weight" in result, prefix
            assert f"{prefix}soft_embedding_norm.weight" in result, prefix

    def test_audio_keys_dropped_when_config_has_no_audio(self):
        """An audio-less package must not be handed audio weights.

        They would only produce "not applied" warnings for a component that is
        absent.
        """
        config = _tiny_multimodal_config(audio=None, audio_token_id=None)
        model = Gemma3nMultiModalModel(config)
        assert model.audio_encoder is None

        result = model.preprocess_weights(
            {
                f"{_HF}audio_tower.conformer.0.norm.weight": torch.zeros(1),
                f"{_HF}embed_audio.embedding.weight": torch.zeros(1),
                f"{_HF}vision_tower.timm_model.msfa.norm.weight": torch.zeros(1),
            }
        )

        assert not any("audio" in key for key in result)
        assert "vision_encoder.encoder.msfa.norm.weight" in result

    def test_kv_shared_layer_weights_are_dropped(self):
        """The checkpoint ships K/V for all layers; shared layers build none."""
        config = _tiny_multimodal_config(
            layer_types=["full_attention", "full_attention"],
            num_kv_shared_layers=1,
        )
        model = Gemma3nMultiModalModel(config)
        assert model.decoder.model.layers[1].self_attn.is_kv_shared_layer

        suffixes = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "k_norm.weight")
        result = model.preprocess_weights(
            {
                f"{_HF}language_model.layers.{idx}.self_attn.{suffix}": torch.zeros(1)
                for idx in (0, 1)
                for suffix in suffixes
            }
        )

        for suffix in ("k_proj.weight", "v_proj.weight", "k_norm.weight"):
            assert f"decoder.model.layers.0.self_attn.{suffix}" in result, suffix
            assert f"decoder.model.layers.1.self_attn.{suffix}" not in result, suffix
        # Q is per-layer and is never shared.
        assert "decoder.model.layers.1.self_attn.q_proj.weight" in result

    def test_unprefixed_keys_pass_through(self):
        """Keys already in ONNX form (e.g. a re-loaded package) survive."""
        model = Gemma3nMultiModalModel(_tiny_multimodal_config())

        result = model.preprocess_weights({"decoder.model.norm.weight": torch.zeros(1)})

        assert "decoder.model.norm.weight" in result


def _build_text_decoder(**overrides):
    """Build the text-only gemma3n causal-LM graph from a tiny config."""
    from mobius._registry import registry
    from mobius.tasks import get_task

    defaults = dict(
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_type="default",
        rope_theta=10_000.0,
        rope_local_base_freq=10_000.0,
    )
    defaults.update(overrides)
    config = _tiny_gemma3n_config(**defaults)
    module = registry.get("gemma3n_text")(config)
    return config, get_task(module.default_task).build(module, config)["model"]


def _fill_random_weights(model: ir.Model) -> None:
    """Assign small random constants to every unfilled initializer."""
    rng = np.random.default_rng(0)
    for value in model.graph.initializers.values():
        if value.const_value is not None:
            continue
        shape = [d if isinstance(d, int) else 1 for d in value.shape]
        value.const_value = ir.tensor(rng.standard_normal(shape).astype(np.float32) * 0.05)


class TestGemma3nKvSharedCausalMasking:
    """KV-shared layers must not use the Attention op's built-in causal mask.

    They pass the borrowed K,V *whole* with no ``past_key``/``past_value``, so
    at decode ``q_len=1`` while ``kv_len=total``.  The opset-24 spec aligns the
    built-in mask upper-left, which pins that single query to key 0.  ORT <=
    1.27 aligned it bottom-right and masked the bug; 1.28 conforms and the
    decode output collapses onto BOS.  Causality has to come from the explicit
    ``attn_mask`` instead.
    """

    @staticmethod
    def _attention_nodes(model: ir.Model):
        """Split Attention nodes by whether they feed the op's KV cache path."""
        with_past, without_past = [], []
        for node in model.graph:
            if node.op_type != "Attention":
                continue
            has_past = len(node.inputs) > 4 and node.inputs[4] is not None
            (with_past if has_past else without_past).append(node)
        return with_past, without_past

    def test_shared_layers_disable_is_causal(self):
        _, model = _build_text_decoder(
            layer_types=["full_attention", "full_attention"],
            num_kv_shared_layers=1,
        )
        with_past, without_past = self._attention_nodes(model)

        assert len(with_past) == 1, "layer 0 owns a cache entry"
        assert len(without_past) == 1, "layer 1 borrows K,V whole"
        assert without_past[0].attributes["is_causal"].as_int() == 0
        assert with_past[0].attributes["is_causal"].as_int() == 1

    def test_no_sharing_keeps_is_causal(self):
        """Without sharing every layer takes the cache path and keeps the flag."""
        _, model = _build_text_decoder(num_kv_shared_layers=0)
        with_past, without_past = self._attention_nodes(model)

        assert not without_past
        assert with_past and all(n.attributes["is_causal"].as_int() == 1 for n in with_past)

    def test_decode_step_matches_full_prefill(self):
        """The invariant the flag broke: cached decode == re-running the prefill.

        Feeding the whole shared K,V with an upper-left causal mask makes the
        decode query attend to position 0 alone, so its logits stop tracking the
        equivalent row of a full forward.  This is prefill-vs-decode, so the
        prefill-only synthetic parity sweep cannot see it.
        """
        config, model = _build_text_decoder(
            num_hidden_layers=3,
            layer_types=["full_attention", "sliding_attention", "sliding_attention"],
            num_kv_shared_layers=1,
            sliding_window=64,
        )
        _fill_random_weights(model)
        session = ort.InferenceSession(
            ir.serde.serialize_model(model).SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        out_names = [o.name for o in session.get_outputs()]
        past_names = [i.name for i in session.get_inputs() if i.name.startswith("past_")]

        rng = np.random.default_rng(0)
        tokens = rng.integers(1, config.vocab_size, size=(1, 5)).astype(np.int64)

        def feed(ids, mask_len, past):
            return {
                "input_ids": ids,
                "attention_mask": np.ones((1, mask_len), dtype=np.int64),
                "position_ids": np.arange(mask_len - ids.shape[1], mask_len, dtype=np.int64)[
                    np.newaxis, :
                ],
                **past,
            }

        empty = {
            name: np.zeros((1, config.num_key_value_heads, 0, config.head_dim), np.float32)
            for name in past_names
        }
        prefill = session.run(None, feed(tokens[:, :4], 4, empty))
        prefill = dict(zip(out_names, prefill))
        cache = {
            name: prefill[name.replace("past_key_values", "present")] for name in past_names
        }
        decode = session.run(None, feed(tokens[:, 4:], 5, cache))[out_names.index("logits")]

        reference = session.run(None, feed(tokens, 5, empty))[out_names.index("logits")]
        np.testing.assert_allclose(decode[:, -1], reference[:, -1], rtol=1e-3, atol=1e-3)


def _hf_key_for(initializer_name: str) -> str:
    """Invert ``preprocess_weights`` for one ONNX initializer name.

    Written independently of the implementation (from the E4B checkpoint's
    ``model.safetensors.index.json`` key layout) so a wrong rename shows up as
    a missing initializer rather than cancelling out.
    """
    tower_prefixes = (
        ("vision_encoder.encoder.", "vision_tower.timm_model."),
        ("audio_encoder.encoder.", "audio_tower."),
        ("vision_encoder.embed_vision.", "embed_vision."),
        ("embedding.embed_vision.", "embed_vision."),
        ("audio_encoder.embed_audio.", "embed_audio."),
        ("embedding.embed_audio.", "embed_audio."),
        ("embedding.", "language_model."),
        ("decoder.lm_head.", "language_model.lm_head."),
    )
    for onnx_prefix, hf_suffix in tower_prefixes:
        if initializer_name.startswith(onnx_prefix):
            return _HF + hf_suffix + initializer_name[len(onnx_prefix) :]

    assert initializer_name.startswith("decoder.model."), initializer_name
    suffix = initializer_name[len("decoder.model.") :]
    parts = suffix.split(".")
    altup_members = {
        "correct_output_scale",
        "correction_coefs",
        "prediction_coefs",
        "modality_router",
        "router_norm",
    }
    if parts[0] == "layers" and parts[2] in altup_members:
        # preprocess_weights strips the ``.altup.`` level; put it back.
        suffix = ".".join([*parts[:2], "altup", *parts[2:]])
    return f"{_HF}language_model.{suffix}"


class TestGemma3nMultiModalWeightCoverage:
    """Every initializer needing data must be filled by a real checkpoint.

    ``ModelPackage.save`` raises when any initializer still has
    ``const_value is None``, so a rename that misses even one tensor turns a
    working export into a hard failure — or, as with the original vision-blind
    gemma3n export, into a package whose towers were never populated.  This
    walks the *built* package and checks the full HF -> ONNX round-trip.
    """

    @staticmethod
    def _unfilled_initializers(pkg) -> dict[str, list[str]]:
        """Map component name -> initializer names still awaiting weights."""
        return {
            name: [
                init_name
                for init_name, init in model.graph.initializers.items()
                if init.const_value is None
            ]
            for name, model in pkg.items()
        }

    def _assert_full_coverage(self, config) -> None:
        from mobius._registry import registry
        from mobius.tasks import get_task

        module = registry.get("gemma3n")(config)
        pkg = get_task("gemma3n").build(module, config)
        unfilled = self._unfilled_initializers(pkg)

        # Synthesize the checkpoint the way HF ships it, then rename.
        state_dict = {}
        for component, names in unfilled.items():
            for name in names:
                shape = list(pkg[component].graph.initializers[name].shape)
                state_dict[_hf_key_for(name)] = torch.ones(shape)
        renamed = module.preprocess_weights(state_dict)

        missing = {
            component: sorted(set(names) - set(renamed))
            for component, names in unfilled.items()
            if set(names) - set(renamed)
        }
        assert not missing, f"preprocess_weights left initializers unfilled: {missing}"

    def test_all_initializers_covered_with_audio(self):
        self._assert_full_coverage(_tiny_multimodal_config())

    def test_all_initializers_covered_without_audio(self):
        self._assert_full_coverage(_tiny_multimodal_config(audio=None, audio_token_id=None))

    def test_all_initializers_covered_with_kv_sharing(self):
        """KV sharing removes initializers; it must not remove needed ones."""
        self._assert_full_coverage(
            _tiny_multimodal_config(
                layer_types=["full_attention", "full_attention"],
                num_kv_shared_layers=1,
            )
        )

    def test_vision_tower_weights_actually_reach_the_graph(self):
        """The regression guard: the MobileNet-V5 tower must be populated.

        The causal-LM ``preprocess_weights`` deleted ``vision_tower.`` keys,
        which produced an 8 GB "multimodal" package that could not see images.
        """
        config = _tiny_multimodal_config()
        from mobius._registry import registry
        from mobius.tasks import get_task

        module = registry.get("gemma3n")(config)
        pkg = get_task("gemma3n").build(module, config)
        vision_names = self._unfilled_initializers(pkg)["vision_encoder"]
        assert len(vision_names) > 500, "MobileNet-V5 tower should be large"

        renamed = module.preprocess_weights(
            {_hf_key_for(name): torch.ones(1) for name in vision_names}
        )

        assert set(vision_names) <= set(renamed)
