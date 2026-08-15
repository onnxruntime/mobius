# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GGUF adapter for NVIDIA Nemotron 3.5 Lightning."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np
import onnx_ir as ir
import torch

from mobius._configs import NemotronHConfig, QuantizationConfig
from mobius.integrations.gguf._architecture import (
    GGUFArchitectureAdapter,
    GGUFMappingAudit,
    GGUFTensorMapping,
    GGUFTensorTarget,
    register_architecture_adapter,
)
from mobius.integrations.gguf._repacker import RepackedTensor
from mobius.integrations.gguf._tensor_processors import _reverse_permute

if TYPE_CHECKING:
    from mobius.integrations.gguf._reader import GGUFModel

_BLOCK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")

_PINNED_LAYER_TYPES = (
    "mamba2",
    "moe",
    "mamba2",
    "moe",
    "mamba2",
    "full_attention",
    "moe",
    "mamba2",
    "moe",
    "mamba2",
    "moe",
    "mamba2",
    "full_attention",
    "moe",
    "mamba2",
    "moe",
    "mamba2",
    "moe",
    "mamba2",
    "full_attention",
    "moe",
    "mamba2",
    "moe",
    "mamba2",
    "moe",
    "mamba2",
    "full_attention",
    "moe",
    "mamba2",
    "moe",
    "mamba2",
    "moe",
    "mamba2",
    "full_attention",
    "moe",
    "mamba2",
    "moe",
    "mamba2",
    "moe",
    "mamba2",
    "moe",
    "mamba2",
    "full_attention",
    "moe",
    "mamba2",
    "moe",
    "mamba2",
    "moe",
    "mamba2",
    "moe",
    "mamba2",
    "moe",
)

_MAMBA_SUFFIXES = frozenset(
    {
        "attn_norm.weight",
        "ssm_a",
        "ssm_conv1d.bias",
        "ssm_conv1d.weight",
        "ssm_d",
        "ssm_dt.bias",
        "ssm_in.weight",
        "ssm_norm.weight",
        "ssm_out.weight",
    }
)
_MOE_SUFFIXES = frozenset(
    {
        "attn_norm.weight",
        "exp_probs_b.bias",
        "ffn_down_exps.weight",
        "ffn_down_shexp.weight",
        "ffn_gate_inp.weight",
        "ffn_up_exps.weight",
        "ffn_up_shexp.weight",
    }
)
_ATTENTION_SUFFIXES = frozenset(
    {
        "attn_k.weight",
        "attn_norm.weight",
        "attn_output.weight",
        "attn_q.weight",
        "attn_v.weight",
    }
)
_Q8_BASE_SUFFIXES = frozenset(
    {
        "attn_k.weight",
        "attn_output.weight",
        "attn_q.weight",
        "attn_v.weight",
        "ffn_down_exps.weight",
        "ffn_down_shexp.weight",
        "ffn_up_exps.weight",
        "ffn_up_shexp.weight",
        "ssm_in.weight",
        "ssm_out.weight",
    }
)
_MTP_SUFFIXES = frozenset(
    {
        "attn_k.weight",
        "attn_norm.weight",
        "attn_output.weight",
        "attn_q.weight",
        "attn_v.weight",
        "exp_probs_b.bias",
        "ffn_down_exps.weight",
        "ffn_down_shexp.weight",
        "ffn_gate_inp.weight",
        "ffn_up_exps.weight",
        "ffn_up_shexp.weight",
        "nextn.eh_proj.weight",
        "nextn.enorm.weight",
        "nextn.hnorm.weight",
        "nextn.shared_head_norm.weight",
        "post_attention_norm.weight",
    }
)
_EXPECTED_SUFFIXES = {
    "mamba2": _MAMBA_SUFFIXES,
    "moe": _MOE_SUFFIXES,
    "full_attention": _ATTENTION_SUFFIXES,
}


def _as_int(value: Any, name: str) -> int:
    if value is None:
        raise ValueError(f"nemotron_h_moe GGUF is missing metadata {name!r}")
    return int(value)


def _qtype_name(qtype: Any) -> str:
    return str(getattr(qtype, "name", qtype))


def _reverse_permute_array(value: np.ndarray, n_head: int) -> np.ndarray:
    dim = value.shape[0] // n_head // 2
    return value.reshape(n_head, dim, 2, *value.shape[1:]).swapaxes(1, 2).reshape(value.shape)


@register_architecture_adapter
class NemotronHMoEAdapter(GGUFArchitectureAdapter):
    """Strict adapter for the pinned Nemotron 3.5 Lightning GGUF layout."""

    architecture = "nemotron_h_moe"
    model_type = "nemotron_h"

    def __init__(self, model: GGUFModel) -> None:
        super().__init__(model)
        self._block_suffixes = self._collect_block_suffixes(model.tensor_names)
        self._mtp_blocks = tuple(
            index
            for index, suffixes in sorted(self._block_suffixes.items())
            if any(suffix.startswith("nextn.") for suffix in suffixes)
        )
        self._layer_types = self._derive_layer_types()
        self._config: NemotronHConfig | None = None

    @staticmethod
    def _collect_block_suffixes(tensor_names: list[str]) -> dict[int, frozenset[str]]:
        result: dict[int, set[str]] = {}
        for name in tensor_names:
            match = _BLOCK_RE.match(name)
            if match is not None:
                result.setdefault(int(match.group(1)), set()).add(match.group(2))
        return {index: frozenset(suffixes) for index, suffixes in result.items()}

    def _derive_layer_types(self) -> tuple[str, ...]:
        layer_types = []
        for block_index, suffixes in sorted(self._block_suffixes.items()):
            if block_index in self._mtp_blocks:
                continue
            kinds = []
            if any(suffix.startswith("ssm_") for suffix in suffixes):
                kinds.append("mamba2")
            if any(suffix.startswith(("ffn_", "exp_probs_")) for suffix in suffixes):
                kinds.append("moe")
            if any(
                suffix.startswith(("attn_q.", "attn_k.", "attn_v.", "attn_output."))
                for suffix in suffixes
            ):
                kinds.append("full_attention")
            if len(kinds) != 1:
                raise ValueError(
                    f"Nemotron backbone block {block_index} has ambiguous mixer types: {kinds}"
                )
            if block_index != len(layer_types):
                raise ValueError(
                    "Nemotron backbone blocks must be contiguous from zero; "
                    f"expected {len(layer_types)}, found {block_index}"
                )
            layer_types.append(kinds[0])
        return tuple(layer_types)

    def validate_model(self, *, source: str) -> None:
        metadata = self.model.metadata
        block_count = _as_int(metadata.get(f"{self.architecture}.block_count"), "block_count")
        if block_count != 53:
            raise ValueError(
                f"Expected 53 Nemotron GGUF blocks, got {block_count} in {source!r}"
            )
        if self._layer_types != _PINNED_LAYER_TYPES:
            counts = Counter(self._layer_types)
            raise ValueError(
                "Nemotron GGUF backbone schedule differs from the pinned 52-layer "
                f"contract: {dict(counts)}"
            )
        if self._mtp_blocks != (52,):
            raise ValueError(
                f"Expected separate combined attention+MoE MTP block 52, got {self._mtp_blocks}"
            )

        for index, layer_type in enumerate(self._layer_types):
            actual = self._block_suffixes[index]
            expected = _EXPECTED_SUFFIXES[layer_type]
            if actual != expected:
                raise ValueError(
                    f"Nemotron block {index} ({layer_type}) tensor inventory mismatch; "
                    f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
                )
        if self._block_suffixes[52] != _MTP_SUFFIXES:
            actual = self._block_suffixes[52]
            raise ValueError(
                "Nemotron MTP block 52 tensor inventory mismatch; "
                f"missing={sorted(_MTP_SUFFIXES - actual)}, "
                f"extra={sorted(actual - _MTP_SUFFIXES)}"
            )

        records = list(self.model._reader.tensors)
        if len(records) != 417:
            raise ValueError(f"Expected 417 pinned Nemotron tensors, got {len(records)}")

        unsupported: Counter[str] = Counter()
        qtype_counts: Counter[str] = Counter()
        base_qtype_counts: Counter[str] = Counter()
        mtp_qtype_counts: Counter[str] = Counter()
        for record in records:
            qtype = _qtype_name(record.tensor_type)
            qtype_counts[qtype] += 1
            shape = tuple(int(dim) for dim in reversed(record.shape))
            is_mtp = record.name.startswith("blk.52.")
            (mtp_qtype_counts if is_mtp else base_qtype_counts)[qtype] += 1
            if not is_mtp:
                match = _BLOCK_RE.match(record.name)
                suffix = match.group(2) if match is not None else None
                expected_qtype = (
                    "Q8_0"
                    if record.name in {"token_embd.weight", "output.weight"}
                    or suffix in _Q8_BASE_SUFFIXES
                    else "F32"
                )
                if qtype in {"F32", "F16", "BF16", "Q8_0"} and qtype != expected_qtype:
                    raise ValueError(
                        f"Nemotron base tensor {record.name!r} has qtype {qtype}, "
                        f"expected {expected_qtype}; exact Q8 preservation does not "
                        "dequantize a quantized source into a float-only target"
                    )
            if not is_mtp and qtype not in {"F32", "F16", "BF16", "Q8_0"}:
                unsupported[qtype] += math.prod(shape)

        if unsupported:
            observed = ", ".join(
                f"{qtype}={parameters:,} parameters"
                for qtype, parameters in sorted(unsupported.items())
            )
            raise NotImplementedError(
                "Nemotron 3.5 GGUF conversion currently preserves the validated "
                f"Q8_0 production slice only; observed unsupported base types: {observed}. "
                "These source formats do not have a validated runtime kernel mapping, "
                "so conversion is refused instead of dequantizing and calling it preservation."
            )

        if base_qtype_counts != Counter({"F32": 237, "Q8_0": 164}):
            raise ValueError(
                "Pinned Nemotron base qtype inventory mismatch: "
                f"{dict(sorted(base_qtype_counts.items()))}"
            )
        if mtp_qtype_counts != Counter({"Q8_0": 9, "F32": 6, "BF16": 1}):
            raise ValueError(
                "Pinned Nemotron MTP qtype inventory mismatch: "
                f"{dict(sorted(mtp_qtype_counts.items()))}"
            )
        if qtype_counts != Counter({"Q8_0": 173, "F32": 243, "BF16": 1}):
            raise ValueError(
                f"Pinned Nemotron total qtype inventory mismatch: {dict(qtype_counts)}"
            )

        tokenizer_model = metadata.get("tokenizer.ggml.model")
        tokenizer_pre = metadata.get("tokenizer.ggml.pre")
        if (tokenizer_model, tokenizer_pre) != ("gpt2", "pixtral"):
            raise ValueError(
                "Nemotron tokenizer contract requires GGUF GPT-2/Pixtral BPE metadata; "
                f"got model={tokenizer_model!r}, pre={tokenizer_pre!r}"
            )
        special_ids = {
            "bos": metadata.get("tokenizer.ggml.bos_token_id"),
            "eos": metadata.get("tokenizer.ggml.eos_token_id"),
            "padding": metadata.get("tokenizer.ggml.padding_token_id"),
        }
        if special_ids != {"bos": 1, "eos": 11, "padding": 999}:
            raise ValueError(
                f"Unexpected pinned Nemotron GGUF special-token ids: {special_ids}"
            )

    def build_config(self) -> NemotronHConfig:
        if self._config is not None:
            return self._config

        metadata = self.model.metadata
        prefix = f"{self.architecture}."
        hidden_size = _as_int(metadata.get(prefix + "embedding_length"), "embedding_length")
        attention_heads = _as_int(
            metadata.get(prefix + "attention.head_count"), "attention.head_count"
        )
        head_dim = _as_int(
            metadata.get(prefix + "attention.key_length"), "attention.key_length"
        )
        kv_values = metadata.get(prefix + "attention.head_count_kv")
        if not isinstance(kv_values, list):
            raise TypeError("nemotron_h_moe attention.head_count_kv must be a per-block list")
        nonzero_kv_heads = {int(value) for value in kv_values if int(value) > 0}
        if len(nonzero_kv_heads) != 1:
            raise ValueError(
                f"Expected one nonzero Nemotron KV-head count, got {sorted(nonzero_kv_heads)}"
            )

        inner_size = _as_int(metadata.get(prefix + "ssm.inner_size"), "ssm.inner_size")
        mamba_heads = _as_int(
            metadata.get(prefix + "ssm.time_step_rank"), "ssm.time_step_rank"
        )
        if inner_size % mamba_heads:
            raise ValueError(
                f"SSM inner size {inner_size} is not divisible by {mamba_heads} Mamba heads"
            )

        self._config = NemotronHConfig(
            vocab_size=_as_int(metadata.get(prefix + "vocab_size"), "vocab_size"),
            hidden_size=hidden_size,
            intermediate_size=_as_int(
                metadata.get(prefix + "expert_feed_forward_length"),
                "expert_feed_forward_length",
            ),
            num_hidden_layers=len(self._layer_types),
            num_attention_heads=attention_heads,
            num_key_value_heads=nonzero_kv_heads.pop(),
            head_dim=head_dim,
            hidden_act="relu2",
            # The GGUF declares EOS=11 and PAD=999, while the pinned runtime
            # contract uses model EOS=2/PAD=0 and accepts 11 as a second
            # generation stop. Tokenizer asset padding remains role token 11.
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=False,
            attn_qkv_bias=False,
            attn_o_bias=False,
            dtype=ir.DataType.FLOAT,
            max_position_embeddings=_as_int(
                metadata.get(prefix + "context_length"), "context_length"
            ),
            layer_types=list(self._layer_types),
            rms_norm_eps=float(
                metadata.get(prefix + "attention.layer_norm_rms_epsilon", 1e-5)
            ),
            rope_type=None,
            rope_theta=None,
            partial_rotary_factor=None,
            mlp_bias=False,
            num_local_experts=_as_int(metadata.get(prefix + "expert_count"), "expert_count"),
            num_experts_per_tok=_as_int(
                metadata.get(prefix + "expert_used_count"), "expert_used_count"
            ),
            moe_intermediate_size=_as_int(
                metadata.get(prefix + "expert_feed_forward_length"),
                "expert_feed_forward_length",
            ),
            shared_expert_intermediate_size=_as_int(
                metadata.get(prefix + "expert_shared_feed_forward_length"),
                "expert_shared_feed_forward_length",
            ),
            norm_topk_prob=bool(metadata.get(prefix + "expert_weights_norm", True)),
            n_group=_as_int(metadata.get(prefix + "expert_group_count"), "expert_group_count"),
            topk_group=_as_int(
                metadata.get(prefix + "expert_group_used_count"), "expert_group_used_count"
            ),
            routed_scaling_factor=float(metadata.get(prefix + "expert_weights_scale", 1.0)),
            scoring_func="sigmoid",
            n_shared_experts=_as_int(
                metadata.get(prefix + "expert_shared_count"), "expert_shared_count"
            ),
            num_nextn_predict_layers=_as_int(
                metadata.get(prefix + "nextn_predict_layers"), "nextn_predict_layers"
            ),
            mamba_n_heads=mamba_heads,
            mamba_d_head=inner_size // mamba_heads,
            mamba_d_state=_as_int(metadata.get(prefix + "ssm.state_size"), "ssm.state_size"),
            mamba_n_groups=_as_int(
                metadata.get(prefix + "ssm.group_count"), "ssm.group_count"
            ),
            mamba_d_conv=_as_int(metadata.get(prefix + "ssm.conv_kernel"), "ssm.conv_kernel"),
            mamba_expand=2,
            mamba_conv_bias=True,
            mamba_proj_bias=False,
            mamba_time_step_min=0.001,
            mamba_ssm_cache_dtype=ir.DataType.FLOAT,
            moe_latent_size=None,
        )
        self._config._gguf_model_type = self.model_type
        self._config.model_type = self.model_type
        return self._config

    def quantization_config(self) -> QuantizationConfig:
        return QuantizationConfig(
            bits=8,
            group_size=32,
            quant_method="gguf",
            sym=False,
            quantize_embeddings=True,
            quantize_lm_head=True,
            tie_word_embeddings=False,
        )

    def _target(
        self,
        state_dict_name: str,
        initializer_name: str,
        *,
        source_index: int | None = None,
    ) -> GGUFTensorMapping:
        return GGUFTensorMapping(
            (
                GGUFTensorTarget(
                    state_dict_name,
                    initializer_name,
                    source_index=source_index,
                ),
            )
        )

    def _validate_source_shape(
        self,
        source_name: str,
        actual: tuple[int, ...],
        expected: tuple[int, ...],
    ) -> None:
        if actual != expected:
            raise ValueError(
                f"Nemotron GGUF tensor {source_name!r} has shape {actual}, expected {expected}"
            )

    def map_tensor(
        self,
        source_name: str,
        source_shape: tuple[int, ...],
    ) -> GGUFTensorMapping | None:
        config = self.build_config()
        h = config.hidden_size
        q = config.num_attention_heads * config.head_dim
        kv = config.num_key_value_heads * config.head_dim
        d_inner = config.mamba_n_heads * config.mamba_d_head
        conv_dim = d_inner + 2 * config.mamba_n_groups * config.mamba_d_state
        experts = config.num_local_experts
        moe_inner = config.moe_intermediate_size
        shared_inner = config.shared_expert_intermediate_size
        assert experts is not None
        assert moe_inner is not None
        assert shared_inner is not None

        global_mapping = {
            "token_embd.weight": (
                "backbone.embeddings.weight",
                "model.embed_tokens.weight",
                (config.vocab_size, h),
            ),
            "output_norm.weight": (
                "backbone.norm_f.weight",
                "model.norm.weight",
                (h,),
            ),
            "output.weight": ("lm_head.weight", "lm_head.weight", (config.vocab_size, h)),
        }
        if source_name in global_mapping:
            state_name, initializer_name, expected_shape = global_mapping[source_name]
            self._validate_source_shape(source_name, source_shape, expected_shape)
            return self._target(state_name, initializer_name)

        match = _BLOCK_RE.match(source_name)
        if match is None:
            return None
        block_index = int(match.group(1))
        suffix = match.group(2)
        if block_index in self._mtp_blocks:
            return GGUFTensorMapping.excluded("auxiliary MTP block outside the decoder graph")
        if block_index >= len(self._layer_types):
            return None

        state_prefix = f"backbone.layers.{block_index}"
        init_prefix = f"model.layers.{block_index}"
        if suffix == "attn_norm.weight":
            self._validate_source_shape(source_name, source_shape, (h,))
            return self._target(
                f"{state_prefix}.norm.weight",
                f"{init_prefix}.norm.weight",
            )

        layer_type = self._layer_types[block_index]
        if layer_type == "mamba2":
            mappings = {
                "ssm_in.weight": (
                    "in_proj.weight",
                    (d_inner + conv_dim + config.mamba_n_heads, h),
                ),
                "ssm_out.weight": ("out_proj.weight", (h, d_inner)),
                "ssm_conv1d.weight": (
                    "conv1d.weight",
                    (conv_dim, config.mamba_d_conv),
                ),
                "ssm_conv1d.bias": ("conv1d.bias", (conv_dim,)),
                "ssm_a": ("A_log", (config.mamba_n_heads, 1)),
                "ssm_d": ("D", (config.mamba_n_heads, 1)),
                "ssm_dt.bias": ("dt_bias", (config.mamba_n_heads,)),
                "ssm_norm.weight": (
                    "norm.weight",
                    (config.mamba_n_groups, d_inner // config.mamba_n_groups),
                ),
            }
            mapped = mappings.get(suffix)
            if mapped is None:
                return None
            target_suffix, expected_shape = mapped
            self._validate_source_shape(source_name, source_shape, expected_shape)
            return self._target(
                f"{state_prefix}.mixer.{target_suffix}",
                f"{init_prefix}.mamba.{target_suffix}",
            )

        if layer_type == "full_attention":
            mappings = {
                "attn_q.weight": ("q_proj.weight", (q, h)),
                "attn_k.weight": ("k_proj.weight", (kv, h)),
                "attn_v.weight": ("v_proj.weight", (kv, h)),
                "attn_output.weight": ("o_proj.weight", (h, q)),
            }
            mapped = mappings.get(suffix)
            if mapped is None:
                return None
            target_suffix, expected_shape = mapped
            self._validate_source_shape(source_name, source_shape, expected_shape)
            return self._target(
                f"{state_prefix}.mixer.{target_suffix}",
                f"{init_prefix}.self_attn.{target_suffix}",
            )

        mappings = {
            "ffn_gate_inp.weight": ("gate.weight", "gate.weight", (experts, h)),
            "exp_probs_b.bias": (
                "gate.e_score_correction_bias",
                "gate.e_score_correction_bias",
                (experts,),
            ),
            "ffn_up_shexp.weight": (
                "shared_experts.up_proj.weight",
                "shared_experts.up_proj.weight",
                (shared_inner, h),
            ),
            "ffn_down_shexp.weight": (
                "shared_experts.down_proj.weight",
                "shared_experts.down_proj.weight",
                (h, shared_inner),
            ),
        }
        mapped = mappings.get(suffix)
        if mapped is not None:
            state_suffix, init_suffix, expected_shape = mapped
            self._validate_source_shape(source_name, source_shape, expected_shape)
            return self._target(
                f"{state_prefix}.mixer.{state_suffix}",
                f"{init_prefix}.moe.{init_suffix}",
            )

        if suffix in {"ffn_up_exps.weight", "ffn_down_exps.weight"}:
            projection = "up_proj" if suffix.startswith("ffn_up") else "down_proj"
            expected_shape = (
                (experts, moe_inner, h) if projection == "up_proj" else (experts, h, moe_inner)
            )
            self._validate_source_shape(source_name, source_shape, expected_shape)
            targets = tuple(
                GGUFTensorTarget(
                    f"{state_prefix}.mixer.experts.{index}.{projection}.weight",
                    f"{init_prefix}.moe.experts.{index}.{projection}.weight",
                    source_index=index,
                )
                for index in range(experts)
            )
            return GGUFTensorMapping(targets)
        return None

    def transform_tensor(
        self,
        source_name: str,
        target: GGUFTensorTarget,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        del target
        if source_name.endswith(".ssm_a"):
            tensor = tensor.squeeze(-1)
            if not torch.all(tensor < 0):
                raise ValueError(f"Nemotron SSM A tensor {source_name!r} must be negative")
            return torch.log(-tensor)
        if source_name.endswith(".ssm_d"):
            return tensor.squeeze(-1)
        if source_name.endswith(".ssm_norm.weight"):
            return tensor.flatten()
        if source_name.endswith(".ssm_conv1d.weight"):
            return tensor.unsqueeze(1)
        if source_name.endswith(".attn_q.weight"):
            return _reverse_permute(tensor, self.build_config().num_attention_heads)
        if source_name.endswith(".attn_k.weight"):
            return _reverse_permute(tensor, self.build_config().num_key_value_heads)
        return tensor

    def transform_repacked(
        self,
        source_name: str,
        target: GGUFTensorTarget,
        tensor: RepackedTensor,
    ) -> RepackedTensor:
        del target
        n_head = None
        if source_name.endswith(".attn_q.weight"):
            n_head = self.build_config().num_attention_heads
        elif source_name.endswith(".attn_k.weight"):
            n_head = self.build_config().num_key_value_heads
        if n_head is None:
            return tensor
        return RepackedTensor(
            weight=_reverse_permute_array(tensor.weight, n_head),
            scales=_reverse_permute_array(tensor.scales, n_head),
            zero_points=(
                None
                if tensor.zero_points is None
                else _reverse_permute_array(tensor.zero_points, n_head)
            ),
            block_size=tensor.block_size,
            bits=tensor.bits,
        )

    def validate_mapping_audit(self, audit: GGUFMappingAudit) -> None:
        super().validate_mapping_audit(audit)
        if len(audit.mapped_sources) != 401:
            raise ValueError(
                f"Expected 401 mapped Nemotron sources, got {len(audit.mapped_sources)}"
            )
        if len(audit.excluded_sources) != 16:
            raise ValueError(
                f"Expected 16 explicit MTP exclusions, got {len(audit.excluded_sources)}"
            )
        if len(audit.target_sources) != 6243:
            raise ValueError(
                f"Expected 6243 logical Nemotron targets, got {len(audit.target_sources)}"
            )
