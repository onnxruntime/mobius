#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Generate the production-derived reduced GLM-5.3 real-weight fixture.

The script range-reads only the exact leading tensor regions needed by a
two-layer, four-expert GLM-5.3 model. Every value comes from the immutable
production checkpoint; no complete 5.3 GB shard is downloaded. It then runs
the pinned Hugging Face implementation to create L4 prefill and L5 cached
generation references.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import requests
import safetensors.torch
import torch

MODEL_ID = "zai-org/GLM-5.3-Flash"
MODEL_REVISION = "03eb5366286afd40d2221b1d9c63a6dd1ba4832e"
TRANSFORMERS_REVISION = "eb4d9e2a64a013bec12289288b85d0b1210ba0aa"
SOURCE_LAYER_MAP = {0: 0, 1: 3}


@dataclass(frozen=True)
class TensorSpan:
    name: str
    shard: str
    dtype: str
    source_shape: tuple[int, ...]
    byte_start: int
    byte_end: int
    payload_sha256: str


class RangeCheckpoint:
    """Pinned safetensors reader that never requests a whole shard."""

    def __init__(self, index_path: Path):
        index = json.loads(index_path.read_text(encoding="utf-8"))
        self.weight_map: dict[str, str] = index["weight_map"]
        self.session = requests.Session()
        self.headers: dict[str, tuple[int, dict]] = {}
        self.evidence: list[TensorSpan] = []

    def _url(self, shard: str) -> str:
        return f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{shard}"

    def _range(self, shard: str, start: int, end: int) -> bytes:
        response = self.session.get(
            self._url(shard),
            headers={"Range": f"bytes={start}-{end}"},
            timeout=120,
        )
        response.raise_for_status()
        if response.status_code != 206 or len(response.content) != end - start + 1:
            raise RuntimeError(
                f"Server did not honor bounded range {start}-{end} for {shard}: "
                f"status={response.status_code}, bytes={len(response.content)}"
            )
        return response.content

    def _header(self, shard: str) -> tuple[int, dict]:
        if shard not in self.headers:
            length = struct.unpack("<Q", self._range(shard, 0, 7))[0]
            raw = self._range(shard, 8, 7 + length)
            self.headers[shard] = (8 + length, json.loads(raw))
        return self.headers[shard]

    @staticmethod
    def _decode(raw: bytes, dtype: str) -> torch.Tensor:
        if dtype == "BF16":
            return torch.frombuffer(bytearray(raw), dtype=torch.uint16).view(torch.bfloat16)
        if dtype == "F32":
            return torch.frombuffer(bytearray(raw), dtype=torch.float32)
        if dtype == "F16":
            return torch.frombuffer(bytearray(raw), dtype=torch.float16)
        if dtype == "F8_E4M3":
            return torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(
                torch.float8_e4m3fn
            )
        raise ValueError(f"Unsupported fixture source dtype: {dtype}")

    @staticmethod
    def _element_size(dtype: str) -> int:
        return {"BF16": 2, "F32": 4, "F16": 2, "F8_E4M3": 1}[dtype]

    def read_leading(self, name: str, target_shape: tuple[int, ...]) -> torch.Tensor:
        shard = self.weight_map[name]
        data_base, header = self._header(shard)
        entry = header[name]
        source_shape = tuple(int(dim) for dim in entry["shape"])
        dtype = entry["dtype"]
        if len(source_shape) != len(target_shape):
            raise ValueError(
                f"Rank mismatch for {name}: source={source_shape}, target={target_shape}"
            )
        if any(target > source for target, source in zip(target_shape, source_shape)):
            raise ValueError(
                f"Target slice exceeds {name}: source={source_shape}, target={target_shape}"
            )

        if not source_shape:
            leading_shape: tuple[int, ...] = ()
            elements = 1
        elif len(source_shape) == 1:
            leading_shape = target_shape
            elements = target_shape[0]
        else:
            leading_shape = (target_shape[0], *source_shape[1:])
            elements = math.prod(leading_shape)
        offset_start, _offset_end = entry["data_offsets"]
        start = data_base + int(offset_start)
        end = start + elements * self._element_size(dtype)
        raw = self._range(shard, start, end - 1)
        self.evidence.append(
            TensorSpan(
                name=name,
                shard=shard,
                dtype=dtype,
                source_shape=source_shape,
                byte_start=start,
                byte_end=end,
                payload_sha256=hashlib.sha256(raw).hexdigest(),
            )
        )
        tensor = self._decode(raw, dtype).reshape(leading_shape)
        if target_shape:
            tensor = tensor[tuple(slice(0, dim) for dim in target_shape)]

        if dtype.startswith("F8"):
            scale_name = f"{name}_scale_inv"
            if name.endswith(".weight"):
                scale_name = f"{name[: -len('.weight')]}.weight_scale_inv"
            target_scale_shape = tuple((dim + 127) // 128 for dim in target_shape)
            scale = self.read_leading(scale_name, target_scale_shape).float()
            result = tensor.float()
            for row in range(target_shape[0]):
                for column in range(target_shape[1]):
                    result[row, column] *= scale[row // 128, column // 128]
            return result
        return tensor.float()


def _hf_config():
    from transformers.models.glm5_next.configuration_glm5_next import (
        Glm5NextConfig,
    )

    return Glm5NextConfig(
        text_config={
            "vocab_size": 40,
            "hidden_size": 16,
            "intermediate_size": 32,
            "moe_intermediate_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "q_lora_rank": 8,
            "kv_lora_rank": 8,
            "qk_nope_head_dim": 4,
            "qk_rope_head_dim": 0,
            "v_head_dim": 4,
            "rms_norm_eps": 1e-5,
            "layer_types": [
                "linear_attention",
                "deepseek_sparse_attention",
            ],
            "mlp_layer_types": ["dense", "sparse"],
            "linear_attn_config": {
                "num_heads": 4,
                "head_dim": 4,
                "short_conv_kernel_size": 2,
                "gate_lower_bound": -5.0,
                "safe_gate": True,
            },
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": True,
            "routed_scaling_factor": 2.5,
            "index_n_heads": 2,
            "index_head_dim": 4,
            "index_topk": 4,
            "index_kpool": 2,
            "indexer_types": ["full", "full"],
            "hc_mult": 2,
            "hc_sinkhorn_iters": 2,
            "hc_eps": 1e-6,
            "swiglu_limit": 10.0,
            "pad_token_id": 0,
            "eos_token_id": [],
            "tie_word_embeddings": False,
            "dtype": "float32",
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 4,
            "in_channels": 3,
            "image_size": 4,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 16,
            "projection_intermediate_size": 32,
            "hidden_act": "silu",
            "swiglu_limit": 10.0,
            "rms_norm_eps": 1e-5,
            "attention_bias": True,
        },
        image_token_id=31,
        video_token_id=32,
        image_start_token_id=33,
        image_end_token_id=34,
        video_start_token_id=35,
        video_end_token_id=36,
        tie_word_embeddings=False,
    )


def _map_layer_name(name: str) -> str:
    for target, source in SOURCE_LAYER_MAP.items():
        marker = f".layers.{target}."
        if marker in name:
            name = name.replace(marker, f".layers.{source}.", 1)
            break
    for site, checkpoint_prefix in (
        ("attn_hc", "hc_attn"),
        ("ffn_hc", "hc_ffn"),
    ):
        for field in ("fn", "base", "scale"):
            name = name.replace(
                f".{site}.{field}",
                f".{checkpoint_prefix}_{field}",
            )
    return name.replace(".self_attn.forget_gate.", ".self_attn.")


def _production_tensor(
    reader: RangeCheckpoint,
    target_name: str,
    target_shape: tuple[int, ...],
) -> torch.Tensor:
    source_name = _map_layer_name(target_name)
    if source_name.endswith(".self_attn.conv1d.weight"):
        prefix = source_name[: -len("conv1d.weight")]
        chunk = target_shape[0] // 3
        return torch.cat(
            [
                reader.read_leading(
                    f"{prefix}{projection}_conv1d.weight",
                    (chunk, *target_shape[1:]),
                )
                for projection in ("q", "k", "v")
            ],
            dim=0,
        )
    if source_name.endswith(".mlp.experts.gate_up_proj"):
        prefix = source_name[: -len("experts.gate_up_proj")]
        experts = []
        intermediate = target_shape[1] // 2
        for expert in range(target_shape[0]):
            gate = reader.read_leading(
                f"{prefix}experts.{expert}.gate_proj.weight",
                (intermediate, target_shape[2]),
            )
            up = reader.read_leading(
                f"{prefix}experts.{expert}.up_proj.weight",
                (intermediate, target_shape[2]),
            )
            experts.append(torch.cat((gate, up), dim=0))
        return torch.stack(experts)
    if source_name.endswith(".mlp.experts.down_proj"):
        prefix = source_name[: -len("experts.down_proj")]
        return torch.stack(
            [
                reader.read_leading(
                    f"{prefix}experts.{expert}.down_proj.weight",
                    target_shape[1:],
                )
                for expert in range(target_shape[0])
            ]
        )
    return reader.read_leading(source_name, target_shape)


def _goldens(model, output_dir: Path) -> None:
    input_ids = torch.tensor([[2, 31, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    pixel_values = torch.arange(4 * 24, dtype=torch.float32).reshape(4, 24) / 96.0
    grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long)
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=grid_thw,
            use_cache=True,
        )
    last_logits = output.logits[0, -1].float()
    top10 = torch.topk(last_logits, 10).indices.tolist()
    l4 = {
        "schema_version": 1,
        "evidence": "reduced-production-derived-real-weights",
        "source_model": MODEL_ID,
        "source_revision": MODEL_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "input_ids": input_ids[0].tolist(),
        "pixel_values_sha256": hashlib.sha256(pixel_values.numpy().tobytes()).hexdigest(),
        "last_logits": last_logits.tolist(),
        "top10_ids": top10,
        "logits_summary": [
            float(last_logits.max()),
            float(last_logits.min()),
            float(last_logits.mean()),
            float(last_logits.std()),
        ],
    }
    (output_dir / "glm5-next-reduced.json").write_text(
        json.dumps(l4, indent=2) + "\n",
        encoding="utf-8",
    )

    generated: list[int] = []
    step_logits: list[list[float]] = []
    past = output.past_key_values
    mask = attention_mask
    next_token = last_logits.argmax().reshape(1, 1)
    with torch.no_grad():
        for _ in range(24):
            generated.append(int(next_token.item()))
            mask = torch.cat((mask, torch.ones((1, 1), dtype=torch.long)), dim=1)
            output = model(
                input_ids=next_token,
                attention_mask=mask,
                past_key_values=past,
                use_cache=True,
            )
            logits = output.logits[0, -1].float()
            step_logits.append(logits.tolist())
            next_token = logits.argmax().reshape(1, 1)
            past = output.past_key_values
    l5 = {
        "schema_version": 1,
        "evidence": "reduced-production-derived-real-weights",
        "source_model": MODEL_ID,
        "source_revision": MODEL_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "generated_tokens": generated,
        "step_logits": step_logits,
        "generated_text": "token-ids-only reduced real-weight reference",
    }
    (output_dir / "glm5-next-reduced_generation.json").write_text(
        json.dumps(l5, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        type=Path,
        required=True,
        help="Pinned model.safetensors.index.json",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(
            "testdata/evidence/vision-language/glm5-next-reduced-real-weights.safetensors"
        ),
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("testdata/evidence/vision-language/glm5-next-reduced-real-weights.json"),
    )
    parser.add_argument(
        "--golden-dir",
        type=Path,
        default=Path("testdata/golden/vision-language"),
    )
    args = parser.parse_args()

    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextForConditionalGeneration,
    )

    config = _hf_config()
    model = Glm5NextForConditionalGeneration(config).float().eval()
    reader = RangeCheckpoint(args.index)
    state = {
        name: _production_tensor(reader, name, tuple(value.shape))
        for name, value in model.state_dict().items()
    }
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Fixture load mismatch: missing={missing}, unexpected={unexpected}"
        )

    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(state, str(args.fixture))
    args.evidence.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "reduced-production-derived-real-weights",
                "source": {
                    "model_id": MODEL_ID,
                    "revision": MODEL_REVISION,
                    "transformers_revision": TRANSFORMERS_REVISION,
                    "selected_decoder_layers": SOURCE_LAYER_MAP,
                },
                "fixture": {
                    "path": str(args.fixture).replace("\\", "/"),
                    "sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
                    "tensor_count": len(state),
                    "bytes": args.fixture.stat().st_size,
                },
                "range_reads": [
                    {
                        "name": item.name,
                        "shard": item.shard,
                        "dtype": item.dtype,
                        "source_shape": item.source_shape,
                        "byte_start": item.byte_start,
                        "byte_end": item.byte_end,
                        "payload_sha256": item.payload_sha256,
                    }
                    for item in reader.evidence
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    args.golden_dir.mkdir(parents=True, exist_ok=True)
    _goldens(model, args.golden_dir)


if __name__ == "__main__":
    main()
