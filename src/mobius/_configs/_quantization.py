# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Weight-quantization configuration parsed from HuggingFace configs."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class QuantizationConfig:
    """Weight quantization parameters parsed from HuggingFace configs.

    Captures the settings from ``quantization_config`` in HuggingFace model
    configs (GPTQ, AWQ, etc.) so models can decide whether to use
    :class:`~mobius.components.QuantizedLinear` instead of
    :class:`~mobius.components.Linear`.
    """

    bits: int = 4
    group_size: int = 128
    quant_method: str = "none"
    sym: bool = True
    # When True, zero_points is a per-block float tensor rather than a
    # bit-packed uint8. Required for codebooks with non-integer offsets
    # (e.g. Tencent SEQ uses 1.5).
    float_zero_point: bool = False
    # When True, the input embedding table is block-wise quantized and is
    # looked up with GatherBlockQuantized instead of a plain Gather. Used by
    # Olive RTN exports and quantized GGUF imports.
    quantize_embeddings: bool = False
    # When True, the LM head projection is block-wise quantized (MatMulNBits).
    # Used by Olive RTN exports and quantized GGUF imports.
    quantize_lm_head: bool = False
    # When True, the input embedding and LM head share one weight table. Olive
    # RTN records this in its own config (``tie_word_embeddings``) and may clear
    # the model's top-level flag, so it is tracked here independently.
    tie_word_embeddings: bool = False
    # Serialized checkpoint format. This distinguishes formats that share the
    # same quant_method but use different tensor layouts.
    format: str | None = None
    # Root-relative Hugging Face module names intentionally kept in floating
    # point by an upstream mixed-precision planner.
    modules_to_not_convert: list[str] | None = None

    @classmethod
    def from_transformers(cls, hf_config) -> QuantizationConfig | None:
        """Parse ``quantization_config`` from a HuggingFace config.

        Returns ``None`` when no quantization is configured.
        """
        qc = getattr(hf_config, "quantization_config", None)
        if qc is None:
            return None
        # qc can be a dict or a HF QuantizationConfig object
        if hasattr(qc, "to_dict"):
            qc = qc.to_dict()
        if not isinstance(qc, dict):
            return None
        method = qc.get("quant_method", "none")
        if method == "none":
            return None
        # FP8 per-tensor quantization (float8_e4m3fn + scalar scale)
        # is handled by dtype casting in _assign_weight(), not by
        # QuantizedLinear block quantization.
        if method == "fp8":
            return None

        if method == "compressed-tensors":
            config_groups = qc.get("config_groups")
            if not isinstance(config_groups, dict) or len(config_groups) != 1:
                raise ValueError(
                    "Mobius currently supports compressed-tensors checkpoints "
                    "with exactly one quantization config group."
                )
            group = next(iter(config_groups.values()))
            if not isinstance(group, dict):
                raise ValueError("Invalid compressed-tensors config group.")
            weights = group.get("weights")
            if not isinstance(weights, dict):
                raise ValueError(
                    "Compressed-tensors config group must define weight quantization."
                )
            if weights.get("type", "int") != "int":
                raise ValueError(
                    "Mobius compressed-tensors import only supports integer weights."
                )
            if (
                group.get("input_activations") is not None
                or group.get("output_activations") is not None
            ):
                raise ValueError(
                    "Mobius compressed-tensors import currently supports weight-only "
                    "quantization; activation quantization is not supported."
                )
            if weights.get("actorder") is not None:
                raise ValueError(
                    "Mobius compressed-tensors import does not support activation-ordered weights."
                )
            if qc.get("kv_cache_scheme") is not None:
                raise ValueError(
                    "Mobius compressed-tensors import does not support quantized KV caches."
                )
            checkpoint_format = group.get("format", qc.get("format"))
            if checkpoint_format != "pack-quantized":
                raise ValueError(
                    "Mobius only supports the compressed-tensors "
                    f"'pack-quantized' format, got {checkpoint_format!r}."
                )
            targets = group.get("targets")
            if targets != ["Linear"]:
                raise ValueError(
                    "Mobius currently supports compressed-tensors groups targeting "
                    f"exactly ['Linear'], got {targets!r}."
                )
            bits = weights.get("num_bits")
            if bits not in (2, 4, 8):
                raise ValueError(
                    "Compressed-tensors MatMulNBits import requires 2, 4, or 8 "
                    f"weight bits, got {bits!r}."
                )
            if not weights.get("symmetric", True):
                raise ValueError(
                    "Asymmetric compressed-tensors checkpoints are not yet supported."
                )
            return cls(
                bits=bits,
                # Some checkpoints, including Gemma 4 W4A16, omit group_size
                # from config.json. It is inferred from weight_scale shapes
                # before graph construction.
                group_size=weights.get("group_size", 0),
                quant_method=method,
                sym=True,
                format=checkpoint_format,
            )

        if method == "quark":
            global_config = qc.get("global_quant_config")
            weights = global_config.get("weight") if isinstance(global_config, dict) else None
            if not isinstance(weights, dict):
                raise ValueError(
                    "Quark quantization_config must define global_quant_config.weight."
                )
            dtype = weights.get("dtype")
            bits_by_dtype = {"int4": 4, "uint4": 4, "int8": 8, "uint8": 8}
            if dtype not in bits_by_dtype:
                raise ValueError(f"Unsupported Quark weight dtype {dtype!r}.")
            group_size = weights.get("group_size")
            if not isinstance(group_size, int) or group_size <= 0:
                raise ValueError(f"Invalid Quark group_size {group_size!r}.")
            return cls(
                bits=bits_by_dtype[dtype],
                group_size=group_size,
                quant_method=method,
                sym=bool(weights.get("symmetric", dtype.startswith("int"))),
                format=(qc.get("export") or {}).get("pack_method"),
                modules_to_not_convert=qc.get("exclude"),
            )

        return cls(
            bits=qc.get("bits", 4),
            group_size=qc.get("group_size", 128),
            quant_method=method,
            sym=qc.get("sym", qc.get("symmetric", True)),
            quantize_embeddings=bool(qc.get("embeds", False)),
            quantize_lm_head=bool(qc.get("lm_head", False)),
            tie_word_embeddings=bool(qc.get("tie_word_embeddings", False)),
            format=qc.get("format"),
            modules_to_not_convert=qc.get("modules_to_not_convert"),
        )
