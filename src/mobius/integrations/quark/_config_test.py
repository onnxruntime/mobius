# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for AMD Quark ``quantization_config`` parsing."""

from __future__ import annotations

import pytest

from mobius._configs import QuantizationConfig


class TestQuarkConfig:
    """Quark nests its layout differently from the GPTQ/AWQ schema."""

    @staticmethod
    def _quark_config(**weight_overrides):
        """A minimal AMD Quark ``quantization_config`` dict."""
        weight = {
            "dtype": "uint4",
            "group_size": 32,
            "qscheme": "per_group",
            "symmetric": False,
            "is_dynamic": False,
            "zero_point_type": "int32",
        }
        weight.update(weight_overrides)
        return {
            "quant_method": "quark",
            "global_quant_config": {"weight": weight},
            "exclude": ["lm_head", "model.vision_tower"],
        }

    def test_quark_reads_the_nested_schema(self):
        """Quark nests its layout, so the generic top-level tail must not run.

        Falling through would silently yield group_size=128 and sym=True
        instead of erroring — a wrong dequantization, not a failure.
        """
        qc = QuantizationConfig.from_value(self._quark_config())

        assert qc.quant_method == "quark"
        assert qc.bits == 4
        assert qc.group_size == 32
        assert qc.sym is False
        assert qc.modules_to_not_convert == ("lm_head", "model.vision_tower")

    def test_quark_symmetric_is_honoured(self):
        qc = QuantizationConfig.from_value(self._quark_config(symmetric=True))

        assert qc.sym is True

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"dtype": "int4"}, "Unsupported Quark weight dtype"),
            ({"dtype": "fp8_e4m3"}, "Unsupported Quark weight dtype"),
            ({"qscheme": "per_tensor"}, "Unsupported Quark qscheme"),
            ({"is_dynamic": True}, "Dynamic Quark weight quantization"),
        ],
    )
    def test_quark_rejects_unsupported_layouts(self, overrides, match):
        """Fail loudly rather than approximating a layout we cannot unpack."""
        with pytest.raises((NotImplementedError, ValueError), match=match):
            QuantizationConfig.from_value(self._quark_config(**overrides))

    def test_quark_rejects_per_layer_overrides(self):
        config = self._quark_config()
        config["layer_quant_config"] = {"model.layers.0": {}}

        with pytest.raises(NotImplementedError, match="layer_quant_config"):
            QuantizationConfig.from_value(config)

    def test_quark_requires_group_size(self):
        with pytest.raises(ValueError, match="group_size"):
            QuantizationConfig.from_value(self._quark_config(group_size=None))

    def test_quark_rejects_layer_type_overrides(self):
        config = self._quark_config()
        config["layer_type_quant_config"] = {"Linear": {}}

        with pytest.raises(NotImplementedError, match="layer_type_quant_config"):
            QuantizationConfig.from_value(config)

    def test_missing_weight_layout_raises(self):
        config = self._quark_config()
        config["global_quant_config"] = {}

        with pytest.raises(ValueError, match=r"missing global_quant_config\.weight"):
            QuantizationConfig.from_value(config)

    def test_non_mapping_weight_layout_raises(self):
        config = self._quark_config()
        config["global_quant_config"] = {"weight": ["uint4"]}

        with pytest.raises(TypeError, match="must be a mapping"):
            QuantizationConfig.from_value(config)

    def test_non_sequence_exclude_raises(self):
        config = self._quark_config()
        config["exclude"] = {"lm_head": True}

        with pytest.raises(TypeError, match="exclude must be a list or tuple"):
            QuantizationConfig.from_value(config)

    def test_regex_exclusions_are_validated(self):
        config = self._quark_config()
        config["exclude"] = ["re:model\\.layers\\.[0-9]+\\.mlp.*"]

        qc = QuantizationConfig.from_value(config)

        assert qc.modules_to_not_convert == ("re:model\\.layers\\.[0-9]+\\.mlp.*",)

    def test_invalid_regex_exclusion_raises(self):
        config = self._quark_config()
        config["exclude"] = ["re:model\\.layers\\.[0-9"]

        with pytest.raises(ValueError, match="Invalid quantization regex"):
            QuantizationConfig.from_value(config)

    def test_excluded_lm_head_is_float(self):
        """Quark marks float modules in ``exclude``, so a listed head is float."""
        qc = QuantizationConfig.from_value(self._quark_config())

        assert qc.quantize_lm_head is False

    def test_lm_head_absent_from_exclude_is_quantized(self):
        """The inverse test that the AWQ/GPTQ reader gets wrong for Quark.

        Quark never writes an explicit ``lm_head`` key, so that parser defaults
        the flag False and a float head is built over packed weights.
        """
        config = self._quark_config()
        config["exclude"] = ["model.vision_tower"]

        qc = QuantizationConfig.from_value(config)

        assert qc.quantize_lm_head is True

    def test_embeddings_are_never_inferred_as_quantized(self):
        """Quark ships embed_tokens as plain BF16, so the flag stays off."""
        config = self._quark_config()
        config["exclude"] = []

        qc = QuantizationConfig.from_value(config)

        assert qc.quantize_embeddings is False
