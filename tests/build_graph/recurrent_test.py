# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Recurrent, state-space, and hybrid attention-SSM L1 tests.

Run the complete L1 suite with ``pytest tests/build_graph``.
"""

from __future__ import annotations

import pytest
from ._support import (
    _assert_outputs_have_shapes_and_dtypes,
    _make_params,
    _run_onnx_checker,
)
from _test_configs import (
    SSM_CONFIGS,
    TINY_HEAD_DIM,
    TINY_HEADS,
    TINY_HIDDEN,
    TINY_INTERMEDIATE,
    TINY_KV_HEADS,
    TINY_LAYERS,
    TINY_VOCAB,
    _base_config,
)

from mobius._registry import registry
from mobius.integrations.transformers._config_resolver import _default_task_for_model
from mobius.tasks import get_task

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
            # Jamba attention is positional-encoding-free.
            rope_type=None,
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

    def test_jamba_transformers_config_uses_exact_schedules(self):
        """Transformers periods resolve without duplicate inherited fields."""
        from transformers import JambaConfig as HFJambaConfig

        from mobius._configs import JambaConfig

        config = JambaConfig.from_transformers(
            HFJambaConfig(
                vocab_size=TINY_VOCAB,
                hidden_size=TINY_HIDDEN,
                intermediate_size=TINY_INTERMEDIATE,
                num_hidden_layers=4,
                num_attention_heads=TINY_HEADS,
                num_key_value_heads=TINY_KV_HEADS,
                attn_layer_period=2,
                attn_layer_offset=1,
                expert_layer_period=2,
                expert_layer_offset=1,
                num_experts=2,
                num_experts_per_tok=1,
                mamba_d_state=8,
                mamba_d_conv=4,
                mamba_expand=2,
                mamba_dt_rank="auto",
            )
        )
        assert config.layer_types == [
            "mamba",
            "full_attention",
            "mamba",
            "full_attention",
        ]
        assert config.expert_layer_indices == [1, 3]
        assert config.mamba_dt_rank == (TINY_HIDDEN + 15) // 16
        assert config.rope_type is None
        assert config.norm_topk_prob is False

    def test_jamba_router_softmaxes_in_float32(self):
        """Jamba routes with a full-expert float32 softmax before top-k."""
        import dataclasses

        import onnx_ir as ir

        from mobius._builder import build_from_module
        from mobius.models.jamba import JambaCausalLMModel
        from mobius.tasks import HybridCausalLMTask

        config = dataclasses.replace(self._jamba_config(), dtype=ir.DataType.FLOAT16)
        model = build_from_module(
            JambaCausalLMModel(config),
            config,
            task=HybridCausalLMTask(),
        )["model"]
        softmaxes = [node for node in model.graph if node.op_type == "Softmax"]
        assert softmaxes
        for softmax in softmaxes:
            assert softmax.inputs[0].producer().op_type == "Cast"

    def test_jamba_low_precision_state_matches_model_dtype(self):
        """The public recurrent ABI matches Transformers cache storage dtype."""
        import dataclasses

        import onnx_ir as ir

        from mobius._builder import build_from_module
        from mobius.models.jamba import JambaCausalLMModel
        from mobius.tasks import HybridCausalLMTask

        config = dataclasses.replace(self._jamba_config(), dtype=ir.DataType.FLOAT16)
        model = build_from_module(
            JambaCausalLMModel(config),
            config,
            task=HybridCausalLMTask(),
        )["model"]
        inputs = {value.name: value for value in model.graph.inputs}
        outputs = {value.name: value for value in model.graph.outputs}
        assert inputs["past_key_values.0.ssm_state"].dtype == ir.DataType.FLOAT16
        assert outputs["present.0.ssm_state"].dtype == ir.DataType.FLOAT16

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

    def test_jamba_preprocesses_fused_experts_in_numeric_order(self):
        """Current Transformers stacks every expert in one fused parameter."""
        import torch

        from mobius.models.jamba import JambaCausalLMModel

        module = JambaCausalLMModel(self._jamba_config())
        gate_up = torch.stack(
            [
                torch.full((2 * TINY_INTERMEDIATE, TINY_HIDDEN), expert + 1.0)
                for expert in range(2)
            ]
        )
        down = torch.stack(
            [torch.full((TINY_HIDDEN, TINY_INTERMEDIATE), expert + 3.0) for expert in range(2)]
        )
        result = module.preprocess_weights(
            {
                "model.layers.1.feed_forward.experts.gate_up_proj.weight": gate_up,
                "model.layers.1.feed_forward.experts.down_proj.weight": down,
            }
        )
        for expert in range(2):
            torch.testing.assert_close(
                result[f"model.layers.1.feed_forward.experts.{expert}.gate_proj.weight"],
                torch.full((TINY_INTERMEDIATE, TINY_HIDDEN), expert + 1.0),
            )
            torch.testing.assert_close(
                result[f"model.layers.1.feed_forward.experts.{expert}.up_proj.weight"],
                torch.full((TINY_INTERMEDIATE, TINY_HIDDEN), expert + 1.0),
            )
            torch.testing.assert_close(
                result[f"model.layers.1.feed_forward.experts.{expert}.down_proj.weight"],
                torch.full((TINY_HIDDEN, TINY_INTERMEDIATE), expert + 3.0),
            )
