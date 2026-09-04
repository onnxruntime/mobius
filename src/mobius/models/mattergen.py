# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Declarative standard-ONNX MatterGen v1.0.3 GemNet-T neural score core.

This module faithfully declares the neural portion of
``mattergen.denoiser.GemNetTDenoiser`` at source commit
``842ffe735f7d06cec89d56aa23d9f001e1124b30`` and is compatible with the
``microsoft/mattergen`` Hub revision
``5244495dd9a979ff71abc7548a0b14b9deb0069a``.

```mermaid
flowchart LR
  H[Host: PBC radius graph and triplets] --> G[edge_index, D_st, V_st, lattice cosines, id_swap, triplet ids]
  C[Host: condition values and unconditional masks] --> P[property encoders]
  T[timestep] --> N[noise_level_encoding]
  A[one-based atomic_numbers] --> E[AtomEmbedding Z - 1]
  N --> Z[latent per crystal]
  P --> Z
  E --> GT[GemNet-T triplet interaction blocks]
  G --> GT
  Z --> GT
  GT --> O[atom logits, Cartesian position score, lattice score, crystal energy]
```

The host owns fractional-to-Cartesian conversion, periodic image/radius graph
construction, symmetric edge ordering, ``id_swap``, sorted triplet indices,
and ``edge_lattice_cosines``. The latter is the source expression
``cosine_similarity(V_st[:, None], cell[batch[edge_index[0]]], dim=-1)`` and
lets this neural core avoid a cell input. In particular ``edge_index[0]`` is
source ``c`` and ``edge_index[1]`` is target ``a``; ``edge_direction`` is
MatterGen's ``V_st = -distance_vec / distance`` convention. The graph
intentionally does not perform element masking, fractional-coordinate
conversion, stochastic sampling, or PBC construction.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius.components import Embedding, Linear
from mobius.integrations.mattergen._configs import MatterGenConditionSpec, MatterGenConfig


def _cast_float(op: OpBuilder, value: ir.Value) -> ir.Value:
    """Cast a value to the source model's float32 basis-computation dtype."""
    return op.Cast(value, to=ir.DataType.FLOAT)


def _scatter_sum(
    op: OpBuilder,
    values: ir.Value,
    indices: ir.Value,
    output_rows: ir.Value,
) -> ir.Value:
    """Sum leading-axis rows into ``[output_rows, *values.shape[1:]]``.

    MatterGen uses ``torch_scatter.scatter(..., reduce="sum")`` for atom,
    structure, and neighbor aggregation.  ``ScatterND(reduction="add")`` is
    the standard-ONNX equivalent and handles repeated atom/edge ids exactly.
    """
    output_shape = op.Concat(output_rows, op.Shape(values, start=1), axis=0)
    initial = op.Expand(op.CastLike(0.0, values), output_shape)
    scatter_indices = op.Unsqueeze(indices, [1])  # (rows, 1)
    return op.ScatterND(initial, scatter_indices, values, reduction="add")


def _ragged_scatter(
    op: OpBuilder,
    values: ir.Value,
    id_reduce: ir.Value,
    id_ragged_idx: ir.Value,
    num_edges: ir.Value,
) -> ir.Value:
    """Materialize MatterGen's dynamically padded triplet tensor.

    ``id_reduce`` is ``id3_ca`` and ``id_ragged_idx`` enumerates neighboring
    ``b -> a`` edges within each ``c -> a`` group.  The zero appended before
    ``ReduceMax`` gives empty-triplet graphs a well-defined one-wide padded
    tensor; all updates remain zero, preserving the source sum semantics.
    """
    safe_ragged = op.Concat(id_ragged_idx, op.Constant(value_ints=[0]), axis=0)
    max_neighbors = op.Add(op.ReduceMax(safe_ragged, keepdims=1), 1)  # (1,)
    padded_shape = op.Concat(num_edges, max_neighbors, op.Shape(values, start=1), axis=0)
    padded = op.Expand(op.CastLike(0.0, values), padded_shape)
    coordinates = op.Concat(
        op.Unsqueeze(id_reduce, [1]),
        op.Unsqueeze(id_ragged_idx, [1]),
        axis=1,
    )  # (triplets, 2)
    return op.ScatterND(padded, coordinates, values)


class _GemNetDense(nn.Module):
    """MatterGen ``Dense`` with source-compatible ``.linear`` parameter path."""

    def __init__(
        self, in_features: int, out_features: int, *, bias: bool = False, silu: bool = False
    ):
        super().__init__()
        # The nested Linear deliberately matches MatterGen Dense:
        # ``<dense>.linear.weight`` / ``<dense>.linear.bias``.
        self.linear = Linear(in_features, out_features, bias=bias)
        self._silu = silu

    def forward(self, op: OpBuilder, value: ir.Value) -> ir.Value:
        value = self.linear(op, value)
        if self._silu:
            # MatterGen ScaledSiLU is SiLU(x) / 0.6, not the usual SiLU.
            value = op.Mul(op.Mul(value, op.Sigmoid(value)), 1.0 / 0.6)
        return value


class _ResidualLayer(nn.Module):
    """GemNet residual MLP: dense layers followed by ``(x + f(x)) / sqrt(2)``."""

    def __init__(self, units: int, *, num_layers: int = 2):
        super().__init__()
        self.dense_mlp = nn.ModuleList(
            [_GemNetDense(units, units, silu=True) for _ in range(num_layers)]
        )

    def forward(self, op: OpBuilder, value: ir.Value) -> ir.Value:
        residual = value
        for layer in self.dense_mlp:
            value = layer(op, value)
        return op.Mul(op.Add(residual, value), 2.0**-0.5)


class _ScalingFactor(nn.Module):
    """Persistent MatterGen ``ScalingFactor.scale_factor`` initializer."""

    def __init__(self):
        super().__init__()
        # Values are loaded from the checkpoint.  Do not bake gemnet-dT.json
        # factors here: later blocks' values are checkpoint-specific.
        self.scale_factor = nn.Parameter([])

    def forward(self, op: OpBuilder, _reference: ir.Value, value: ir.Value) -> ir.Value:
        return op.Mul(value, self.scale_factor)


class _GaussianSmearing(nn.Module):
    """PyG 2.6 GaussianSmearing with its persisted ``offset`` buffer."""

    def __init__(self, num_gaussians: int):
        super().__init__()
        self.offset = nn.Parameter([num_gaussians])
        # PyG stores this scalar as Python state rather than a state-dict key.
        self._coefficient = -0.5 / (1.0 / (num_gaussians - 1)) ** 2

    def forward(self, op: OpBuilder, distance_scaled: ir.Value) -> ir.Value:
        offset = op.CastLike(self.offset, distance_scaled)
        centered = op.Sub(op.Unsqueeze(distance_scaled, [1]), op.Unsqueeze(offset, [0]))
        return op.Exp(op.Mul(op.Mul(centered, centered), self._coefficient))


class _PolynomialEnvelope(nn.Module):
    """MatterGen fifth-order polynomial radial cutoff envelope."""

    def __init__(self, exponent: int = 5):
        super().__init__()
        self._p = exponent
        self._a = -(exponent + 1) * (exponent + 2) / 2
        self._b = exponent * (exponent + 2)
        self._c = -exponent * (exponent + 1) / 2

    def forward(self, op: OpBuilder, distance_scaled: ir.Value) -> ir.Value:
        d_p = op.Pow(distance_scaled, self._p)
        envelope = op.Add(
            1.0,
            op.Add(
                op.Mul(self._a, d_p),
                op.Add(
                    op.Mul(self._b, op.Pow(distance_scaled, self._p + 1)),
                    op.Mul(self._c, op.Pow(distance_scaled, self._p + 2)),
                ),
            ),
        )
        return op.Where(
            op.Less(distance_scaled, 1.0),
            envelope,
            op.CastLike(0.0, distance_scaled),
        )


class _RadialBasis(nn.Module):
    """Gaussian radial basis times the source polynomial envelope."""

    def __init__(self, num_radial: int, cutoff: float):
        super().__init__()
        self.rbf = _GaussianSmearing(num_radial)
        self.envelope = _PolynomialEnvelope()
        self._inv_cutoff = 1.0 / cutoff

    def forward(self, op: OpBuilder, distance: ir.Value) -> ir.Value:
        distance_scaled = op.Mul(distance, self._inv_cutoff)
        envelope = self.envelope(op, distance_scaled)  # (edges,)
        return op.Mul(op.Unsqueeze(envelope, [1]), self.rbf(op, distance_scaled))


class _CircularBasis(nn.Module):
    """Efficient circular basis: radial Gaussian features and Y_l0(cos(phi))."""

    def __init__(self, num_spherical: int, num_radial: int, cutoff: float):
        super().__init__()
        self.radial_basis = _RadialBasis(num_radial, cutoff)
        self._num_spherical = num_spherical

    def forward(
        self,
        op: OpBuilder,
        distance: ir.Value,
        cosine: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        radial = op.Unsqueeze(self.radial_basis(op, distance), [0])  # (1, edges, radial)
        # MatterGen's SymPy-generated basis is real spherical harmonics with
        # m=0: sqrt((2l+1)/(4*pi)) * P_l(cos(phi)), l=0..num_spherical-1.
        p_previous = op.Add(op.Mul(cosine, 0.0), 1.0)
        polynomials = [p_previous]
        if self._num_spherical > 1:
            p_current = cosine
            polynomials.append(p_current)
            for degree in range(2, self._num_spherical):
                p_next = op.Mul(
                    1.0 / degree,
                    op.Sub(
                        op.Mul((2 * degree - 1), op.Mul(cosine, p_current)),
                        op.Mul(degree - 1, p_previous),
                    ),
                )
                polynomials.append(p_next)
                p_previous, p_current = p_current, p_next
        harmonics = [
            op.Mul(math.sqrt((2 * degree + 1) / (4 * math.pi)), polynomial)
            for degree, polynomial in enumerate(polynomials)
        ]
        spherical = op.Concat(*[op.Unsqueeze(value, [1]) for value in harmonics], axis=1)
        return radial, spherical  # (1, E, R), (T, S)


class _EfficientInteractionDownProjection(nn.Module):
    """Source ``EfficientInteractionDownProjection`` with dynamic ragged padding."""

    def __init__(self, num_spherical: int, num_radial: int, emb_size_interm: int):
        super().__init__()
        self.weight = nn.Parameter([num_spherical, num_radial, emb_size_interm])

    def forward(
        self,
        op: OpBuilder,
        radial: ir.Value,
        spherical: ir.Value,
        id_ca: ir.Value,
        id_ragged_idx: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        # Broadcasted MatMul reproduces torch.matmul([1,E,R], [S,R,C])
        # -> [S,E,C], then permutes to (E, C, S).
        radial_weighted = op.Transpose(op.MatMul(radial, self.weight), perm=[1, 2, 0])
        num_edges = op.Shape(radial_weighted, start=0, end=1)
        padded_spherical = _ragged_scatter(op, spherical, id_ca, id_ragged_idx, num_edges)
        return radial_weighted, op.Transpose(padded_spherical, perm=[0, 2, 1])


class _EfficientInteractionBilinear(nn.Module):
    """Source bilinear triplet aggregation with standard-ONNX ScatterND."""

    def __init__(self, emb_size: int, emb_size_interm: int, units_out: int):
        super().__init__()
        self.weight = nn.Parameter([emb_size, emb_size_interm, units_out])

    def forward(
        self,
        op: OpBuilder,
        basis: tuple[ir.Value, ir.Value],
        messages: ir.Value,
        id_reduce: ir.Value,
        id_ragged_idx: ir.Value,
    ) -> ir.Value:
        radial_weighted, spherical = basis  # (E, C, S), (E, S, K)
        num_edges = op.Shape(radial_weighted, start=0, end=1)
        padded_messages = _ragged_scatter(op, messages, id_reduce, id_ragged_idx, num_edges)
        # First contract neighbor K then spherical S: (E,S,K) @ (E,K,D).
        summed_neighbors = op.MatMul(spherical, padded_messages)  # (E, S, D)
        radial_messages = op.MatMul(radial_weighted, summed_neighbors)  # (E, C, D)
        # Batch dimension D selects the corresponding bilinear weight [D,C,O].
        projected = op.MatMul(op.Transpose(radial_messages, perm=[2, 0, 1]), self.weight)
        return op.ReduceSum(projected, [0], keepdims=0)  # (E, O)


class _AtomEmbedding(nn.Module):
    """Atom type table with MatterGen's one-based ``Z - 1`` lookup."""

    def __init__(self, num_atom_types: int, emb_size: int):
        super().__init__()
        self.embeddings = Embedding(num_atom_types, emb_size)

    def forward(self, op: OpBuilder, atomic_numbers: ir.Value) -> ir.Value:
        return self.embeddings(op, op.Sub(atomic_numbers, 1))


class _EdgeEmbedding(nn.Module):
    """Concatenate source/target atom features and radial features into edges."""

    def __init__(self, atom_features: int, edge_features: int, out_features: int):
        super().__init__()
        self.dense = _GemNetDense(
            2 * atom_features + edge_features,
            out_features,
            silu=True,
        )

    def forward(
        self,
        op: OpBuilder,
        atoms: ir.Value,
        edge_features: ir.Value,
        idx_s: ir.Value,
        idx_t: ir.Value,
    ) -> ir.Value:
        source_atoms = op.Gather(atoms, idx_s)  # (E, atom_dim)
        target_atoms = op.Gather(atoms, idx_t)  # (E, atom_dim)
        return self.dense(op, op.Concat(source_atoms, target_atoms, edge_features, axis=-1))


class _AtomUpdateBlock(nn.Module):
    """Aggregate radial-filtered edge messages into atom embeddings."""

    def __init__(self, config: MatterGenConfig):
        super().__init__()
        self.dense_rbf = _GemNetDense(config.emb_size_rbf, config.emb_size_edge)
        self.scale_sum = _ScalingFactor()
        self.layers = nn.ModuleList(
            [
                _GemNetDense(config.emb_size_edge, config.emb_size_atom, silu=True),
                *[_ResidualLayer(config.emb_size_atom) for _ in range(config.num_atom)],
            ]
        )

    def forward(
        self,
        op: OpBuilder,
        atoms: ir.Value,
        messages: ir.Value,
        radial: ir.Value,
        idx_t: ir.Value,
    ) -> ir.Value:
        radial_message = self.dense_rbf(op, radial)  # (E, edge_dim)
        aggregated = _scatter_sum(
            op,
            op.Mul(messages, radial_message),
            idx_t,
            op.Shape(atoms, start=0, end=1),
        )  # (N, edge_dim)
        value = self.scale_sum(op, messages, aggregated)
        for layer in self.layers:
            value = layer(op, value)
        return value  # (N, atom_dim)


class _OutputBlock(_AtomUpdateBlock):
    """GemNet output block with source-compatible energy and direct-force heads."""

    def __init__(self, config: MatterGenConfig):
        super().__init__(config)
        self.out_energy = _GemNetDense(config.emb_size_atom, config.num_targets)
        self.scale_rbf_F = _ScalingFactor()
        self.seq_forces = nn.ModuleList(
            [
                _GemNetDense(config.emb_size_edge, config.emb_size_edge, silu=True),
                *[_ResidualLayer(config.emb_size_edge) for _ in range(config.num_atom)],
            ]
        )
        self.out_forces = _GemNetDense(config.emb_size_edge, config.num_targets)
        self.dense_rbf_F = _GemNetDense(config.emb_size_rbf, config.emb_size_edge)

    def forward(
        self,
        op: OpBuilder,
        atoms: ir.Value,
        messages: ir.Value,
        radial: ir.Value,
        idx_t: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        # The inherited path is the source energy head's ``seq_energy`` alias.
        # This declaration uses ``layers`` once; preprocessing canonicalizes the
        # duplicated PyTorch state-dict alias ``seq_energy`` onto it.
        radial_energy = self.dense_rbf(op, radial)
        energy_hidden = _scatter_sum(
            op,
            op.Mul(messages, radial_energy),
            idx_t,
            op.Shape(atoms, start=0, end=1),
        )
        energy_hidden = self.scale_sum(op, messages, energy_hidden)
        for layer in self.layers:
            energy_hidden = layer(op, energy_hidden)
        energy = self.out_energy(op, energy_hidden)  # (N, 1)

        force_hidden = messages
        for layer in self.seq_forces:
            force_hidden = layer(op, force_hidden)
        force_hidden = op.Mul(force_hidden, self.dense_rbf_F(op, radial))
        force_hidden = self.scale_rbf_F(op, messages, force_hidden)
        return energy, self.out_forces(op, force_hidden)  # (N,1), (E,1)


class _TripletInteraction(nn.Module):
    """GemNet-T triplet message: radial filtering, bilinear angle sum, edge swap."""

    def __init__(self, config: MatterGenConfig):
        super().__init__()
        self.dense_ba = _GemNetDense(config.emb_size_edge, config.emb_size_edge, silu=True)
        self.mlp_rbf = _GemNetDense(config.emb_size_rbf, config.emb_size_edge)
        self.scale_rbf = _ScalingFactor()
        self.mlp_cbf = _EfficientInteractionBilinear(
            config.emb_size_trip,
            config.emb_size_cbf,
            config.emb_size_bil_trip,
        )
        self.scale_cbf_sum = _ScalingFactor()
        self.down_projection = _GemNetDense(
            config.emb_size_edge,
            config.emb_size_trip,
            silu=True,
        )
        self.up_projection_ca = _GemNetDense(
            config.emb_size_bil_trip,
            config.emb_size_edge,
            silu=True,
        )
        self.up_projection_ac = _GemNetDense(
            config.emb_size_bil_trip,
            config.emb_size_edge,
            silu=True,
        )

    def forward(
        self,
        op: OpBuilder,
        messages: ir.Value,
        radial: ir.Value,
        circular: tuple[ir.Value, ir.Value],
        id_ragged_idx: ir.Value,
        id_swap: ir.Value,
        id_ba: ir.Value,
        id_ca: ir.Value,
    ) -> ir.Value:
        incoming = self.dense_ba(op, messages)
        radial_message = op.Mul(incoming, self.mlp_rbf(op, radial))
        incoming = self.scale_rbf(op, incoming, radial_message)
        incoming = self.down_projection(op, incoming)  # (E, trip_dim)
        triplet_messages = op.Gather(incoming, id_ba)  # (T, trip_dim)
        aggregated = self.mlp_cbf(op, circular, triplet_messages, id_ca, id_ragged_idx)
        aggregated = self.scale_cbf_sum(op, triplet_messages, aggregated)
        forward = self.up_projection_ca(op, aggregated)
        reverse = op.Gather(self.up_projection_ac(op, aggregated), id_swap)
        return op.Mul(op.Add(forward, reverse), 2.0**-0.5)


class _InteractionBlockTripletsOnly(nn.Module):
    """One source GemNet-T triplet-only interaction block."""

    def __init__(self, config: MatterGenConfig):
        super().__init__()
        self.dense_ca = _GemNetDense(config.emb_size_edge, config.emb_size_edge, silu=True)
        self.trip_interaction = _TripletInteraction(config)
        self.layers_before_skip = nn.ModuleList(
            [_ResidualLayer(config.emb_size_edge) for _ in range(config.num_before_skip)]
        )
        self.layers_after_skip = nn.ModuleList(
            [_ResidualLayer(config.emb_size_edge) for _ in range(config.num_after_skip)]
        )
        self.atom_update = _AtomUpdateBlock(config)
        self.concat_layer = _EdgeEmbedding(
            config.emb_size_atom,
            config.emb_size_edge,
            config.emb_size_edge,
        )
        self.residual_m = nn.ModuleList(
            [_ResidualLayer(config.emb_size_edge) for _ in range(config.num_concat)]
        )

    def forward(
        self,
        op: OpBuilder,
        atoms: ir.Value,
        messages: ir.Value,
        radial_triplet: ir.Value,
        circular: tuple[ir.Value, ir.Value],
        id_ragged_idx: ir.Value,
        id_swap: ir.Value,
        id_ba: ir.Value,
        id_ca: ir.Value,
        radial_atom: ir.Value,
        idx_s: ir.Value,
        idx_t: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        update = op.Add(
            self.dense_ca(op, messages),
            self.trip_interaction(
                op,
                messages,
                radial_triplet,
                circular,
                id_ragged_idx,
                id_swap,
                id_ba,
                id_ca,
            ),
        )
        update = op.Mul(update, 2.0**-0.5)
        for layer in self.layers_before_skip:
            update = layer(op, update)

        messages = op.Mul(op.Add(messages, update), 2.0**-0.5)
        for layer in self.layers_after_skip:
            messages = layer(op, messages)

        atom_update = self.atom_update(op, atoms, messages, radial_atom, idx_t)
        atoms = op.Mul(op.Add(atoms, atom_update), 2.0**-0.5)
        edge_update = self.concat_layer(op, atoms, messages, idx_s, idx_t)
        for layer in self.residual_m:
            edge_update = layer(op, edge_update)
        return atoms, op.Mul(op.Add(messages, edge_update), 2.0**-0.5)


class _RBFBasedLatticeUpdateBlock(nn.Module):
    """MatterGen direct lattice-score head from radial edge scores."""

    def __init__(self, config: MatterGenConfig):
        super().__init__()
        self.mlp = nn.ModuleList(
            [
                _GemNetDense(config.emb_size_edge, config.emb_size_edge, silu=True),
                _GemNetDense(config.emb_size_edge, config.emb_size_edge),
            ]
        )
        self.dense_rbf_F = _GemNetDense(config.emb_size_rbf, config.emb_size_edge)
        self.out_forces = _GemNetDense(config.emb_size_edge, config.num_targets)

    def forward(
        self,
        op: OpBuilder,
        edge_embeddings: ir.Value,
        edge_direction: ir.Value,
        batch_edge: ir.Value,
        batch_size: ir.Value,
        radial: ir.Value,
    ) -> ir.Value:
        score = edge_embeddings
        for layer in self.mlp:
            score = layer(op, score)
        score = self.out_forces(op, op.Mul(score, self.dense_rbf_F(op, radial)))  # (E,1)

        # Source normalizes each edge score by the number of source edges in its
        # crystal before the symmetric outer-product lattice aggregation.
        edge_count = _scatter_sum(
            op,
            op.Expand(op.CastLike(1.0, score), op.Shape(batch_edge)),
            batch_edge,
            batch_size,
        )
        score = op.Div(score, op.Unsqueeze(op.Gather(edge_count, batch_edge), [1]))

        # ``distance_vec`` in source is V_st * D_st, then normalized again;
        # normalize edge_direction here to preserve that exact dataflow.
        norm = op.Sqrt(op.ReduceSum(op.Mul(edge_direction, edge_direction), [1], keepdims=1))
        unit_direction = op.Div(edge_direction, norm)
        outer = op.Mul(
            op.Unsqueeze(unit_direction, [2]),
            op.Unsqueeze(unit_direction, [1]),
        )  # (E, 3, 3)
        lattice = _scatter_sum(
            op,
            op.Mul(op.Unsqueeze(score, [2]), outer),
            batch_edge,
            batch_size,
        )
        # The reference transposes after scatter; the outer product is symmetric
        # but retain it so exported graph semantics mirror the source literally.
        return op.Transpose(lattice, perm=[0, 2, 1])


class _AngleEdgeEmbedding(nn.Module):
    """Source ``nn.Sequential(Linear, ReLU, Linear)`` with 0/2 key indices."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        # ``nn.ModuleList`` cannot represent Sequential's parameterless ReLU
        # at index 1. Register the two real children under their source keys.
        setattr(self, "0", Linear(input_size, hidden_size))
        setattr(self, "2", Linear(hidden_size, hidden_size))

    def forward(self, op: OpBuilder, value: ir.Value) -> ir.Value:
        first = getattr(self, "0")(op, value)
        return getattr(self, "2")(op, op.Relu(first))


class _GemNetT(nn.Module):
    """MatterGen GemNet-T backbone with triplet interactions and all score heads."""

    def __init__(self, config: MatterGenConfig):
        super().__init__()
        self.radial_basis = _RadialBasis(config.num_radial, config.cutoff)
        self.cbf_basis3 = _CircularBasis(
            config.num_spherical, config.num_radial, config.cutoff
        )
        self.lattice_out_blocks = nn.ModuleList(
            [_RBFBasedLatticeUpdateBlock(config) for _ in range(config.num_blocks + 1)]
        )
        self.mlp_rbf_lattice = _GemNetDense(config.num_radial, config.emb_size_rbf)
        self.mlp_rbf3 = _GemNetDense(config.num_radial, config.emb_size_rbf)
        self.mlp_cbf3 = _EfficientInteractionDownProjection(
            config.num_spherical,
            config.num_radial,
            config.emb_size_cbf,
        )
        self.mlp_rbf_h = _GemNetDense(config.num_radial, config.emb_size_rbf)
        self.mlp_rbf_out = _GemNetDense(config.num_radial, config.emb_size_rbf)
        self.atom_emb = _AtomEmbedding(config.num_atom_types, config.hidden_size)
        self.atom_latent_emb = Linear(
            config.hidden_size + config.latent_dim, config.emb_size_atom
        )
        self.edge_emb = _EdgeEmbedding(
            config.emb_size_atom, config.num_radial, config.emb_size_edge
        )
        self.angle_edge_emb = _AngleEdgeEmbedding(
            config.emb_size_edge + 3,
            config.emb_size_edge,
        )
        self.int_blocks = nn.ModuleList(
            [_InteractionBlockTripletsOnly(config) for _ in range(config.num_blocks)]
        )
        self.out_blocks = nn.ModuleList(
            [_OutputBlock(config) for _ in range(config.num_blocks + 1)]
        )
        self.cond_adapt_layers: _ConditionAdaptLayers | None = None
        self.cond_mixin_layers: _ConditionMixinLayers | None = None
        self._config = config

    def forward(
        self,
        op: OpBuilder,
        atomic_numbers: ir.Value,
        batch: ir.Value,
        latent: ir.Value,
        edge_index: ir.Value,
        edge_distance: ir.Value,
        edge_direction: ir.Value,
        edge_lattice_cosines: ir.Value,
        id_swap: ir.Value,
        id3_ba: ir.Value,
        id3_ca: ir.Value,
        id3_ragged_idx: ir.Value,
        adapter_embeddings: Mapping[str, ir.Value] | None = None,
        adapter_use_unconditional: Mapping[str, ir.Value] | None = None,
        *,
        return_energy: bool = False,
    ) -> tuple[ir.Value, ir.Value, ir.Value] | tuple[ir.Value, ir.Value, ir.Value, ir.Value]:
        idx_s = op.Gather(edge_index, op.Constant(value_int=0), axis=0)
        idx_t = op.Gather(edge_index, op.Constant(value_int=1), axis=0)
        batch_edge = op.Gather(batch, idx_s)
        batch_size = op.Shape(latent, start=0, end=1)

        # Triplet angle of b -> a <- c uses the host's already normalized V_st.
        cosine = op.Clip(
            op.ReduceSum(
                op.Mul(op.Gather(edge_direction, id3_ca), op.Gather(edge_direction, id3_ba)),
                [1],
                keepdims=0,
            ),
            -1.0,
            1.0,
        )
        radial_circular, spherical = self.cbf_basis3(op, edge_distance, cosine)
        radial = self.radial_basis(op, edge_distance)  # (E, num_radial)

        atoms = self.atom_emb(op, atomic_numbers)  # (N, hidden_dim)
        latent_per_atom = op.Gather(latent, batch)  # (N, latent_dim)
        atoms = self.atom_latent_emb(op, op.Concat(atoms, latent_per_atom, axis=1))
        messages = self.edge_emb(op, atoms, radial, idx_s, idx_t)

        # Host-precomputed source cosine_similarity(V_st[:,None], cell[batch_edge]).
        # Each edge has its alignment to the three lattice-vector rows: (E, 3).
        messages = op.Concat(
            messages,
            edge_lattice_cosines,
            axis=-1,
        )
        messages = self.angle_edge_emb(op, messages)

        radial_triplet = self.mlp_rbf3(op, radial)
        circular = self.mlp_cbf3(op, radial_circular, spherical, id3_ca, id3_ragged_idx)
        radial_atom = self.mlp_rbf_h(op, radial)
        radial_output = self.mlp_rbf_out(op, radial)

        energy, edge_force = self.out_blocks[0](op, atoms, messages, radial_output, idx_t)
        radial_lattice = self.mlp_rbf_lattice(op, radial)
        lattice_score = self.lattice_out_blocks[0](
            op,
            messages,
            edge_direction,
            batch_edge,
            batch_size,
            radial_lattice,
        )

        for index, block in enumerate(self.int_blocks):
            if self._config.condition_on_adapt:
                if (
                    adapter_embeddings is None
                    or adapter_use_unconditional is None
                    or self.cond_adapt_layers is None
                    or self.cond_mixin_layers is None
                ):
                    raise RuntimeError("MatterGen adapter layers require condition embeddings and masks.")
                adaptation = op.Mul(atoms, 0.0)
                for condition_name in self._config.condition_on_adapt:
                    condition = adapter_embeddings[condition_name]
                    condition_per_atom = op.Gather(condition, batch)  # (N, hidden_dim)
                    adapted = self.cond_adapt_layers(
                        op,
                        condition_name,
                        index,
                        op.Concat(atoms, condition_per_atom, axis=-1),
                    )
                    adapted = self.cond_mixin_layers(op, condition_name, index, adapted)
                    use_conditional = op.Not(
                        op.Gather(adapter_use_unconditional[condition_name], batch)
                    )
                    adaptation = op.Add(
                        adaptation,
                        op.Mul(
                            op.Unsqueeze(op.CastLike(use_conditional, adapted), [1]), adapted
                        ),
                    )
                atoms = op.Add(atoms, adaptation)

            atoms, messages = block(
                op,
                atoms,
                messages,
                radial_triplet,
                circular,
                id3_ragged_idx,
                id_swap,
                id3_ba,
                id3_ca,
                radial_atom,
                idx_s,
                idx_t,
            )
            block_energy, block_force = self.out_blocks[index + 1](
                op, atoms, messages, radial_output, idx_t
            )
            energy = op.Add(energy, block_energy)
            edge_force = op.Add(edge_force, block_force)
            lattice_score = op.Add(
                lattice_score,
                self.lattice_out_blocks[index + 1](
                    op,
                    messages,
                    edge_direction,
                    batch_edge,
                    batch_size,
                    self.mlp_rbf_lattice(op, radial),
                ),
            )

        # Each scalar edge force is mapped onto V_st and summed at its target a.
        position_score = _scatter_sum(
            op,
            op.Mul(op.Unsqueeze(edge_force, [2]), op.Unsqueeze(edge_direction, [1])),
            idx_t,
            op.Shape(atomic_numbers, start=0, end=1),
        )
        position_score = op.Squeeze(position_score, [1])  # (N, 3)
        if return_energy:
            return atoms, position_score, lattice_score, energy
        return atoms, position_score, lattice_score


class _ScalarNoiseLevelEncoding(nn.Module):
    """MatterGen's interleaved sin/cos ``NoiseLevelEncoding``."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.div_term = nn.Parameter([hidden_dim // 2])
        # Source registers div_term as a float32 buffer.  It must not be
        # demoted with model weights when the exporter requests fp16/bf16.
        setattr(self.div_term, "_keep_float32", True)
        self._hidden_dim = hidden_dim

    def forward(self, op: OpBuilder, value: ir.Value) -> ir.Value:
        value = op.Reshape(_cast_float(op, value), [-1, 1])
        div_term = op.Reshape(_cast_float(op, self.div_term), [1, -1])
        angles = op.Mul(value, div_term)  # (B, hidden_dim / 2)
        # Source writes sin into even and cos into odd columns.  Stack then
        # reshape interleaves them; concatenating sin/cos would be incorrect.
        interleaved = op.Concat(
            op.Unsqueeze(op.Sin(angles), [2]),
            op.Unsqueeze(op.Cos(angles), [2]),
            axis=2,
        )
        return op.Reshape(interleaved, [-1, self._hidden_dim])


class _StandardScaler(nn.Module):
    """Persistent ``StandardScalerTorch`` parameters used before scalar encoding."""

    def __init__(self, *, log10_transform: bool):
        super().__init__()
        self.means = nn.Parameter([1])
        self.stds = nn.Parameter([1])
        self._log10_transform = log10_transform

    def forward(self, op: OpBuilder, value: ir.Value) -> ir.Value:
        value = _cast_float(op, value)
        if self._log10_transform:
            value = op.Div(op.Log(value), math.log(10.0))
        return op.Div(op.Sub(value, self.means), self.stds)


class _EmbeddingVector(nn.Module):
    """Source unconditional condition embedding: Embedding(1, hidden_dim)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.embedding = Embedding(1, hidden_dim)

    def forward(self, op: OpBuilder, reference: ir.Value) -> ir.Value:
        target_shape = op.Concat(
            op.Shape(reference, start=0, end=1),
            op.Shape(self.embedding.weight, start=1),
            axis=0,
        )
        return op.Expand(self.embedding.weight, target_shape)


class _ChemicalSystemMultiHotEmbedding(nn.Module):
    """Source chemical-system multi-hot linear encoder."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.embedding = Linear(101, hidden_dim)

    def forward(self, op: OpBuilder, value: ir.Value) -> ir.Value:
        return self.embedding(op, value)


class _SpaceGroupEmbeddingVector(nn.Module):
    """Source one-based space-group encoder: gather ``embedding[x.long() - 1]``."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.embedding = Embedding(230, hidden_dim)

    def forward(self, op: OpBuilder, value: ir.Value) -> ir.Value:
        return self.embedding(op, op.Sub(value, 1))


class _PropertyEmbedding(nn.Module):
    """Faithful source property switch between conditional and unconditional embeddings."""

    def __init__(self, spec: MatterGenConditionSpec, hidden_dim: int):
        super().__init__()
        self._spec = spec
        self.conditional_embedding_module: (
            _ScalarNoiseLevelEncoding
            | _ChemicalSystemMultiHotEmbedding
            | _SpaceGroupEmbeddingVector
        )
        if spec.kind == "scalar_sinusoidal":
            self.conditional_embedding_module = _ScalarNoiseLevelEncoding(hidden_dim)
        elif spec.kind == "chemical_system_multihot":
            self.conditional_embedding_module = _ChemicalSystemMultiHotEmbedding(hidden_dim)
        elif spec.kind == "space_group_index":
            self.conditional_embedding_module = _SpaceGroupEmbeddingVector(hidden_dim)
        else:  # MatterGenConfig.validate already rejects this path.
            raise ValueError(f"Unsupported MatterGen condition encoder {spec.kind!r}")

        if spec.unconditional == "embedding_vector":
            self.unconditional_embedding_module = _EmbeddingVector(hidden_dim)
        elif spec.unconditional != "zeros":
            raise ValueError(
                f"Unsupported unconditional condition embedding {spec.unconditional!r}"
            )
        if spec.scaler == "standard":
            self.scaler = _StandardScaler(log10_transform=spec.log10_transform)

    def forward(
        self,
        op: OpBuilder,
        value: ir.Value,
        use_unconditional: ir.Value,
        dtype: ir.DataType,
    ) -> ir.Value:
        if self._spec.scaler == "standard":
            value = self.scaler(op, value)
        conditional = self.conditional_embedding_module(op, value)
        conditional = op.Cast(conditional, to=dtype)
        if self._spec.unconditional == "zeros":
            unconditional = op.Expand(op.CastLike(0.0, conditional), op.Shape(conditional))
        else:
            unconditional = self.unconditional_embedding_module(op, conditional)
        mask = op.Reshape(use_unconditional, [-1, 1])
        return op.Where(mask, unconditional, conditional)


class _NamedConditionModules(nn.Module):
    """Small ModuleDict equivalent preserving MatterGen condition-name paths."""

    def __init__(self, specs: Sequence[MatterGenConditionSpec], hidden_dim: int):
        super().__init__()
        self._names = tuple(spec.name for spec in specs)
        for spec in specs:
            setattr(self, spec.name, _PropertyEmbedding(spec, hidden_dim))

    def get(self, name: str) -> _PropertyEmbedding:
        return getattr(self, name)

    def forward(
        self,
        op: OpBuilder,
        name: str,
        value: ir.Value,
        use_unconditional: ir.Value,
        dtype: ir.DataType,
    ) -> ir.Value:
        """Invoke the named child beneath this container's ONNX name scope."""
        return self.get(name)(op, value, use_unconditional, dtype)


class _ConditionAdaptLayers(nn.Module):
    """Source ``cond_adapt_layers.<condition>.<block>`` ModuleDict hierarchy."""

    def __init__(self, condition_names: Sequence[str], config: MatterGenConfig):
        super().__init__()
        self._names = tuple(condition_names)
        for name in condition_names:
            layers = nn.ModuleList(
                [_ConditionAdaptLayer(config.hidden_size) for _ in range(config.num_blocks)]
            )
            setattr(self, name, layers)

    def forward(
        self,
        op: OpBuilder,
        name: str,
        block_index: int,
        value: ir.Value,
    ) -> ir.Value:
        """Apply one named source adapter MLP in its declared container scope."""
        return getattr(self, name)[block_index](op, value)


class _ConditionAdaptLayer(nn.Module):
    """Adapter MLP preserving PyTorch Sequential's ``.0`` and ``.2`` paths."""

    def __init__(self, hidden_size: int):
        super().__init__()
        setattr(self, "0", Linear(hidden_size * 2, hidden_size))
        setattr(self, "2", Linear(hidden_size, hidden_size))

    def forward(self, op: OpBuilder, value: ir.Value) -> ir.Value:
        first = getattr(self, "0")(op, value)
        return getattr(self, "2")(op, op.Relu(first))


class _ConditionMixinLayers(nn.Module):
    """Source ``cond_mixin_layers.<condition>.<block>`` zero-init linear hierarchy."""

    def __init__(self, condition_names: Sequence[str], config: MatterGenConfig):
        super().__init__()
        self._names = tuple(condition_names)
        for name in condition_names:
            setattr(
                self,
                name,
                nn.ModuleList(
                    [
                        Linear(config.hidden_size, config.hidden_size, bias=False)
                        for _ in range(config.num_blocks)
                    ]
                ),
            )

    def forward(
        self,
        op: OpBuilder,
        name: str,
        block_index: int,
        value: ir.Value,
    ) -> ir.Value:
        """Apply one named source mixin linear layer in its declared scope."""
        return getattr(self, name)[block_index](op, value)


class MatterGenModel(nn.Module):
    """MatterGen GemNet-T score core with explicit host-provided geometric tensors.

    .. mermaid::

       flowchart LR
           H[Host: noisy crystal and periodic graph] --> G[ONNX: GemNet-T score core]
           C[Host: condition values and CFG masks] --> G
           G --> S[Host: source scheduler and crystal validation]

    ``edge_lattice_cosines`` is host-produced ``[E,3]`` from the source's
    lattice cosine calculation; this preserves GemNet's angle-edge embedding
    without accepting a cell tensor. ``condition_values`` and
    ``condition_use_unconditional`` are mappings keyed by
    :attr:`MatterGenConfig.condition_input_specs` names. Each value is ``[B]``
    for scalar/space-group properties or ``[B,101]`` for chemical systems;
    every mask is bool ``[B]`` where true selects the unconditional embedding.
    A later MatterGen task uses this config-derived ABI to declare named ONNX
    ports. The final ``energy`` output retains the trained GemNet OutputBlock
    energy path, so every official checkpoint initializer remains reachable.
    """

    default_task: str = "mattergen-score"
    category: str = "Diffusion"
    config_class = MatterGenConfig

    def __init__(self, config: MatterGenConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.noise_level_encoding = _ScalarNoiseLevelEncoding(config.hidden_size)
        self.property_embeddings = _NamedConditionModules(
            config.property_embeddings,
            config.hidden_size,
        )
        self.property_embeddings_adapt = _NamedConditionModules(
            config.property_embeddings_adapt,
            config.hidden_size,
        )
        self.gemnet = _GemNetT(config)
        if config.condition_on_adapt:
            # These are attributes of the GemNet source module, not the
            # denoiser, and therefore live beneath ``gemnet`` in state dicts.
            self.gemnet.cond_adapt_layers = _ConditionAdaptLayers(
                config.condition_on_adapt, config
            )
            self.gemnet.cond_mixin_layers = _ConditionMixinLayers(
                config.condition_on_adapt, config
            )
        self.fc_atom = Linear(config.hidden_size, config.num_atom_types)

    def forward(
        self,
        op: OpBuilder,
        atomic_numbers: ir.Value,
        batch: ir.Value,
        timestep: ir.Value,
        edge_index: ir.Value,
        edge_distance: ir.Value,
        edge_direction: ir.Value,
        edge_lattice_cosines: ir.Value,
        id_swap: ir.Value,
        id3_ba: ir.Value,
        id3_ca: ir.Value,
        id3_ragged_idx: ir.Value,
        condition_values: Mapping[str, ir.Value] | None = None,
        condition_use_unconditional: Mapping[str, ir.Value] | None = None,
    ) -> tuple[ir.Value, ir.Value, ir.Value, ir.Value]:
        """Return atom logits, position score, lattice score, and crystal energy.

        The outputs are respectively ``[N,101]``, ``[N,3]``, ``[B,3,3]``,
        and source-aggregated ``[B,1]`` energy.
        """
        specs = self.config.condition_input_specs
        condition_values = {} if condition_values is None else condition_values
        condition_use_unconditional = (
            {} if condition_use_unconditional is None else condition_use_unconditional
        )
        expected_condition_names = {spec.name for spec in specs}
        if (
            set(condition_values) != expected_condition_names
            or set(condition_use_unconditional) != expected_condition_names
        ):
            raise ValueError(
                "MatterGen condition value and mask mappings must each contain "
                "exactly the config.condition_input_specs names"
            )

        timestep_embedding = self.noise_level_encoding(op, timestep)  # (B, hidden_dim)
        base_embeddings: list[ir.Value] = []
        adapter_embeddings: dict[str, ir.Value] = {}
        adapter_masks: dict[str, ir.Value] = {}
        for spec in specs:
            collection = (
                self.property_embeddings
                if not spec.is_adapter
                else self.property_embeddings_adapt
            )
            embedding = collection(
                op,
                spec.name,
                condition_values[spec.name],
                condition_use_unconditional[spec.name],
                self.config.dtype,
            )
            if not spec.is_adapter:
                base_embeddings.append(embedding)
            else:
                adapter_embeddings[spec.name] = embedding
                adapter_masks[spec.name] = condition_use_unconditional[spec.name]

        latent = op.Cast(timestep_embedding, to=self.config.dtype)
        if base_embeddings:
            # Source's get_property_embeddings sorts ModuleDict keys, which is
            # encoded by MatterGenConfig before the module is constructed.
            latent = op.Concat(latent, *base_embeddings, axis=-1)

        atom_embeddings, position_score, lattice_score, atom_energy = self.gemnet(
            op,
            atomic_numbers,
            batch,
            latent,
            edge_index,
            edge_distance,
            edge_direction,
            edge_lattice_cosines,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
            adapter_embeddings,
            adapter_masks,
            return_energy=True,
        )
        # Source applies fc_atom to the final GemNet atom embedding, not to
        # GemNet's separate auxiliary per-atom energy estimate.
        atom_logits = self.fc_atom(op, atom_embeddings)
        # GemNet sums each per-atom target within its crystal before returning
        # ModelOutput.energy. ``timestep`` explicitly carries the batch width.
        energy = _scatter_sum(
            op,
            atom_energy,
            batch,
            op.Shape(timestep, start=0, end=1),
        )
        return atom_logits, position_score, lattice_score, energy

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Strip the Lightning model prefix and validate exact score-core routing.

        Only tensors rooted at ``diffusion_module.model.`` (or an already
        stripped inference mapping) are accepted; no training keys are
        discarded here. PyTorch serializes OutputBlock's shared ``layers``
        module twice under ``seq_energy``; that documented alias is
        canonicalized after verifying its duplicate tensor agrees.
        """
        prefix = "diffusion_module.model."
        routed: dict[str, torch.Tensor] = {}
        for source_name, value in state_dict.items():
            if source_name.startswith(prefix):
                name = source_name.removeprefix(prefix)
            elif source_name.startswith(
                (
                    "noise_level_encoding.",
                    "fc_atom.",
                    "property_embeddings.",
                    "property_embeddings_adapt.",
                    "gemnet.",
                )
            ):
                # Already stripped state dictionaries are accepted unchanged.
                name = source_name
            else:
                raise ValueError(f"Unexpected MatterGen checkpoint key: {source_name!r}")

            canonical = name.replace(".seq_energy.", ".layers.")
            existing = routed.get(canonical)
            if existing is not None:
                if not torch.equal(existing, value):
                    raise ValueError(
                        f"Conflicting MatterGen OutputBlock alias tensors for {canonical!r}"
                    )
                continue
            routed[canonical] = value

        expected = set(self.state_dict())
        unexpected = set(routed) - expected
        if unexpected:
            raise ValueError(f"Unexpected routed MatterGen parameters: {sorted(unexpected)!r}")
        missing = expected - set(routed)
        if missing:
            raise KeyError(f"Missing MatterGen score-core parameters: {sorted(missing)!r}")
        return routed


# A descriptive public alias for callers that need to distinguish this score
# core from host-side MatterGen sampling orchestration.
MatterGenGemNetTModel = MatterGenModel

__all__ = ["MatterGenGemNetTModel", "MatterGenModel"]
