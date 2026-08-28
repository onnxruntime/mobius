# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Fixed-resolution SigLIP tower plus an exact-GELU two-layer projector.

This module models the reusable closure used by fixed-resolution ``clip``
sidecars whose vision tower already emits final, post-layernorm patch states::

    pixels -> SigLIP -> Linear(Dv, Dp) -> GELU(exact) -> Linear(Dp, Dt)

The tower is injected so callers can reuse the existing SigLIP implementation
without duplicating it. For Janus-Pro, the shapes are ``(B, 576, 1024)`` after
the tower and ``(B, 576, 2048)`` after both projector layers.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from onnxscript import OpBuilder, nn

from mobius.components._common import Linear

if TYPE_CHECKING:
    import onnx_ir as ir


class ExactGELUMLPProjector(nn.Module):
    """Two affine layers separated by the erf-based, non-approximate GELU.

    Inputs have shape ``(..., vision_hidden_size)`` and outputs have shape
    ``(..., output_hidden_size)``. ``hidden_size`` is the intermediate width.
    """

    def __init__(
        self,
        vision_hidden_size: int,
        hidden_size: int,
        output_hidden_size: int | None = None,
    ):
        super().__init__()
        if min(vision_hidden_size, hidden_size, output_hidden_size or hidden_size) <= 0:
            raise ValueError("Projector dimensions must be positive")
        output_hidden_size = output_hidden_size or hidden_size
        self.linear_0 = Linear(vision_hidden_size, hidden_size, bias=True)
        self.linear_1 = Linear(hidden_size, output_hidden_size, bias=True)

    def forward(self, op: OpBuilder, vision_features: ir.Value):
        # (..., Dv) -> (..., Dp); no ``approximate`` attribute means exact GELU.
        hidden_states = op.Gelu(self.linear_0(op, vision_features))
        return self.linear_1(op, hidden_states)  # (..., Dt)


class FixedResolutionSiglipEmbeddings(nn.Module):
    """Adapt existing SigLIP embeddings to a fixed, fully populated patch grid.

    Fixed sidecars always consume every learned position in raster order, so
    the ``(patches, hidden)`` table broadcasts directly over the batch. This
    avoids constructing dynamic position IDs while preserving the existing
    ``patch_embedding`` and ``position_embedding.weight`` parameter names.
    """

    def __init__(self, patch_embedding: nn.Module, position_embedding: nn.Module):
        super().__init__()
        self.patch_embedding = patch_embedding
        self.position_embedding = position_embedding
        self._num_positions = int(position_embedding.weight.shape[0])  # type: ignore[attr-defined]

    def forward(self, op: OpBuilder, pixel_values: ir.Value):
        patch_states = self.patch_embedding(op, pixel_values)
        position_ids = op.Constant(value_ints=list(range(self._num_positions)))
        return op.Add(patch_states, self.position_embedding(op, position_ids))


class FixedResolutionSiglipMLPSidecar(nn.Module):
    """Compose a fixed-resolution SigLIP tower with an exact-GELU projector.

    ``vision_tower`` must return final patch states without a class token. The
    injected module remains named ``vision_tower`` so existing SigLIP weight
    loading can be reused unchanged.
    """

    def __init__(
        self,
        vision_tower: nn.Module,
        vision_hidden_size: int,
        projector_hidden_size: int,
        output_hidden_size: int | None = None,
    ):
        super().__init__()
        embeddings = vision_tower.embeddings  # type: ignore[attr-defined]
        vision_tower.embeddings = FixedResolutionSiglipEmbeddings(
            embeddings.patch_embedding,
            embeddings.position_embedding,
        )
        self.vision_tower = vision_tower
        self.projector = ExactGELUMLPProjector(
            vision_hidden_size,
            projector_hidden_size,
            output_hidden_size,
        )

    def forward(self, op: OpBuilder, pixel_values: ir.Value):
        # (B, C, H, W) -> (B, (H/P)*(W/P), Dv) -> (B, patches, Dt).
        pixel_values = op.CastLike(
            pixel_values,
            self.vision_tower.embeddings.patch_embedding.projection.weight,  # type: ignore[attr-defined]
        )
        patch_states = self.vision_tower(op, pixel_values)
        return self.projector(op, patch_states)


_BLOCK_WEIGHT = re.compile(r"^v\.blk\.(\d+)\.(.+)$")
_BLOCK_STEMS = {
    "ln1.weight": "layer_norm1.weight",
    "ln1.bias": "layer_norm1.bias",
    "ln2.weight": "layer_norm2.weight",
    "ln2.bias": "layer_norm2.bias",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_out.weight": "self_attn.out_proj.weight",
    "attn_out.bias": "self_attn.out_proj.bias",
    # Standard ViT naming: ``up`` expands Dv -> Dff and ``down`` contracts it.
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_up.bias": "mlp.up_proj.bias",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_down.bias": "mlp.down_proj.bias",
}


def map_fixed_siglip_sidecar_weight(name: str) -> str | None:
    """Map one standard split-QKV fixed-SigLIP sidecar tensor to this module.

    The source vocabulary follows llama.cpp ``clip`` sidecars. Unknown tensors
    return ``None`` rather than being assigned to an approximate topology.
    """
    block = _BLOCK_WEIGHT.fullmatch(name)
    if block is not None:
        layer, suffix = block.groups()
        mapped = _BLOCK_STEMS.get(suffix)
        return None if mapped is None else f"vision_tower.encoder.{layer}.{mapped}"

    return {
        "v.patch_embd.weight": "vision_tower.embeddings.patch_embedding.projection.weight",
        "v.patch_embd.bias": "vision_tower.embeddings.patch_embedding.projection.bias",
        "v.position_embd.weight": "vision_tower.embeddings.position_embedding.weight",
        "v.post_ln.weight": "vision_tower.post_layernorm.weight",
        "v.post_ln.bias": "vision_tower.post_layernorm.bias",
        "mm.0.weight": "projector.linear_0.weight",
        "mm.0.bias": "projector.linear_0.bias",
        "mm.1.weight": "projector.linear_1.weight",
        "mm.1.bias": "projector.linear_1.bias",
    }.get(name)
