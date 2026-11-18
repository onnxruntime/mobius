# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Google GNM v3 articulated head mesh generation.

Replicates ``gnm.shape.gnm_pytorch.GNM``: identity and expression coefficients,
axis-angle joint rotations, and translation produce all mesh vertices. The
ONNX graph includes identity-dependent joints, pose correctives, hierarchical
forward kinematics, and linear blend skinning, without tracing PyTorch.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._configs import BaseModelConfig

GNM_MODEL_ID = "google/gnm-v3"
GNM_REVISION = "c01e90d298d82301f9fd18f54806be751775cb7c"
GNM_SHA256 = "61d78bbfb4ad8e0b38495804a4caef3214d3df00f8c3f68761e63b41ce3747eb"


@dataclasses.dataclass
class GNMConfig(BaseModelConfig):
    """Dimensions and skeleton topology from the native GNM NPZ checkpoint."""

    model_type: str = "gnm"
    num_vertices: int = 17821
    identity_dim: int = 253
    expression_dim: int = 383
    joint_parent_indices: tuple[int, ...] = (-1, 0, 1, 1)

    @classmethod
    def from_transformers(cls, config) -> GNMConfig:
        """Adapt validated native config for registry-driven tooling."""
        if not isinstance(config, cls):
            raise TypeError(
                "GNM has no Transformers config.json; read dimensions from its native "
                "v3_0/gnm_head.npz with GNMConfig.from_pretrained()."
            )
        return dataclasses.replace(config)

    @classmethod
    def from_pretrained(cls, model_id: str, *, revision: str | None = None) -> GNMConfig:
        """Read dimensions from the same pinned native archive used for weights."""
        from mobius.integrations._gnm import gnm_archive_path, read_gnm_config

        with np.load(gnm_archive_path(model_id, revision), allow_pickle=False) as archive:
            return read_gnm_config(archive)

    def validate(self) -> None:
        if self.dtype != ir.DataType.FLOAT:
            raise ValueError("GNM supports float32 only; select dtype='f32'.")
        if min(self.num_vertices, self.identity_dim, self.expression_dim) <= 0:
            raise ValueError(
                "GNM vertex, identity and expression dimensions must be positive."
            )
        parents = self.joint_parent_indices
        if not parents or parents[0] not in (-1, 0):
            raise ValueError("GNM requires a root joint with parent -1 or 0.")
        if any(parent < 0 or parent >= i for i, parent in enumerate(parents[1:], 1)):
            raise ValueError("GNM joints must follow their parents in topological order.")
        if self.quantization is not None or self.component_quantization is not None:
            raise ValueError("GNM quantization is not qualified; use float32 weights.")


def _rotation_matrices(op: OpBuilder, rotations: ir.Value) -> ir.Value:
    # Match the upstream epsilon inside sqrt, including the zero-angle branch.
    angle = op.Sqrt(op.Max(op.ReduceSum(op.Mul(rotations, rotations), [-1], keepdims=1), 1e-8))
    axis = op.Div(rotations, angle)
    x, y, z = (op.Gather(axis, [i], axis=-1) for i in range(3))
    zero = op.Mul(x, 0.0)
    skew = op.Reshape(
        op.Concat(zero, op.Neg(z), y, z, zero, op.Neg(x), op.Neg(y), x, zero, axis=-1),
        [0, 0, 3, 3],
    )  # (B, J, 3, 3)
    theta = op.Unsqueeze(angle, [-1])
    eye = op.Constant(value=ir.tensor(np.eye(3, dtype=np.float32)))
    return op.Add(
        op.Add(eye, op.Mul(op.Sin(theta), skew)),
        op.Mul(op.Sub(1.0, op.Cos(theta)), op.MatMul(skew, skew)),
    )


class GNMHeadModel(nn.Module):
    """Google GNM v3 full articulated head, matching upstream PyTorch GNM."""

    default_task = "mesh-generation"
    category = "3D"
    config_class = GNMConfig

    def __init__(self, config: GNMConfig):
        super().__init__()
        config.validate()
        self.config = config
        v, i, e = config.num_vertices, config.identity_dim, config.expression_dim
        j = len(config.joint_parent_indices)
        self.template_vertex_positions = nn.Parameter([v, 3])
        self.template_joint_positions = nn.Parameter([j, 3])
        self.vertex_identity_basis = nn.Parameter([i, v, 3])
        self.joint_identity_basis = nn.Parameter([i, j, 3])
        self.expression_basis = nn.Parameter([e, v, 3])
        self.pose_correctives_regressor = nn.Parameter([j * 9, v * 3])
        self.skinning_weights = nn.Parameter([j, v])

    def forward(self, op, identity, expression, rotations, translation):
        self.config.validate()
        vertices = op.Add(
            op.Add(
                self.template_vertex_positions,
                op.Einsum(identity, self.vertex_identity_basis, equation="bi,ivc->bvc"),
            ),
            op.Einsum(expression, self.expression_basis, equation="be,evc->bvc"),
        )  # (B, V, 3)
        joints = op.Add(
            self.template_joint_positions,
            op.Einsum(identity, self.joint_identity_basis, equation="bi,ijc->bjc"),
        )  # (B, J, 3)
        matrices = _rotation_matrices(op, rotations)
        eye = op.Constant(value=ir.tensor(np.eye(3, dtype=np.float32)))
        features = op.Reshape(op.Sub(matrices, eye), [0, -1])
        correctives = op.Reshape(
            op.MatMul(features, self.pose_correctives_regressor),
            [0, self.config.num_vertices, 3],
        )
        vertices = op.Add(vertices, correctives)

        # Compose rigid transforms along the skeleton, then remove bind joints
        # before blending so identity-dependent eye/neck pivots remain correct.
        world_rotations, world_translations = [], []
        for index, parent in enumerate(self.config.joint_parent_indices):
            rotation = op.Gather(matrices, index, axis=1)  # (B, 3, 3)
            joint = op.Gather(joints, index, axis=1)  # (B, 3)
            if index == 0:
                position = op.Add(joint, translation)
            else:
                offset = op.Sub(joint, op.Gather(joints, parent, axis=1))
                position = op.Add(
                    op.Squeeze(
                        op.MatMul(world_rotations[parent], op.Unsqueeze(offset, [-1])),
                        [-1],
                    ),
                    world_translations[parent],
                )
                rotation = op.MatMul(world_rotations[parent], rotation)
            world_rotations.append(rotation)
            world_translations.append(position)
        world_r = op.Concat(*(op.Unsqueeze(r, [1]) for r in world_rotations), axis=1)
        world_t = op.Concat(*(op.Unsqueeze(t, [1]) for t in world_translations), axis=1)
        bind_offset = op.Squeeze(op.MatMul(world_r, op.Unsqueeze(joints, [-1])), [-1])
        vertex_r = op.Einsum(self.skinning_weights, world_r, equation="jv,bjmn->bvmn")
        vertex_t = op.Einsum(
            self.skinning_weights, op.Sub(world_t, bind_offset), equation="jv,bjm->bvm"
        )
        return op.Add(
            op.Squeeze(op.MatMul(vertex_r, op.Unsqueeze(vertices, [-1])), [-1]),
            vertex_t,
        )  # (B, V, 3)
