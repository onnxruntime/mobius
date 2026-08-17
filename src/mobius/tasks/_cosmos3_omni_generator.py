# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Task for the NVIDIA Cosmos3-Omni unified MoT transformer.

Builds a **single** ONNX graph performing one rectified-flow denoising step of
:class:`~mobius.models.cosmos3_omni_generator.Cosmos3OmniGeneratorModel`.

Packed I/O contract
===================

Upstream's ``Cosmos3OmniTransformer.forward`` takes Python lists of ragged
per-item tensors and per-item ``(T, H, W)`` shape tuples.  Neither can cross an
ONNX boundary, so the graph is cut at the **packed-token boundary**: the host
patchifies/packs latents before ``proj_in`` and unpatchifies/unpacks
predictions after ``proj_out``.  Every tensor operation in between — including
the noisy-row offsets that upstream derives from ``token_shapes`` and the
``und_len`` split — is an explicit tensor input, so no semantics are lost.

Symbolic dimensions
-------------------

===========================  ==========================================
``sequence_length``          rows of the packed joint sequence
``num_text_tokens``          text tokens written into the joint sequence
``num_vision_tokens``        packed vision patch tokens
``num_vision_noisy_tokens``  vision patch tokens carrying noise
``num_sound_tokens``         packed sound frames        (``sound_gen``)
``num_sound_noisy_tokens``   noisy sound frames         (``sound_gen``)
``num_action_tokens``        packed action steps        (``action_gen``)
``num_action_noisy_tokens``  noisy action steps         (``action_gen``)
===========================  ==========================================

Inputs (``D`` = ``config.dtype``, the model compute dtype)
----------------------------------------------------------

===============================  =======  ===============================================
``input_ids``                    int64    ``[num_text_tokens]``
``text_indexes``                 int64    ``[num_text_tokens]`` joint rows for the text
``position_ids``                 int64    ``[3, sequence_length]`` mRoPE (T, H, W)
``und_len``                      int64    ``[1]`` leading rows on the understanding expert
``vision_tokens``                D        ``[num_vision_tokens, patch_latent_dim]``
``vision_sequence_indexes``      int64    ``[num_vision_tokens]`` joint rows
``vision_timesteps``             float32  ``[num_vision_noisy_tokens]``
``vision_timestep_token_indexes``int64    ``[num_vision_noisy_tokens]`` rows of ``vision_tokens``
``vision_mse_loss_indexes``      int64    ``[num_vision_noisy_tokens]`` joint rows to decode
===============================  =======  ===============================================

Present only when ``config.sound_gen``:

===============================  =======  ===============================================
``sound_tokens``                 D        ``[num_sound_tokens, sound_dim]``
``sound_sequence_indexes``       int64    ``[num_sound_tokens]``
``sound_timesteps``              float32  ``[num_sound_noisy_tokens]``
``sound_timestep_token_indexes`` int64    ``[num_sound_noisy_tokens]``
``sound_mse_loss_indexes``       int64    ``[num_sound_noisy_tokens]``
===============================  =======  ===============================================

Present only when ``config.action_gen``:

================================  =======  ==============================================
``action_tokens``                 D        ``[num_action_tokens, action_dim]``
``action_domain_ids``             int64    ``[num_action_tokens]`` embodiment per token
``action_sequence_indexes``       int64    ``[num_action_tokens]``
``action_timesteps``              float32  ``[num_action_noisy_tokens]``
``action_timestep_token_indexes`` int64    ``[num_action_noisy_tokens]``
``action_mse_loss_indexes``       int64    ``[num_action_noisy_tokens]``
``action_pred_domain_ids``        int64    ``[num_action_noisy_tokens]`` embodiment per pred
================================  =======  ==============================================

Outputs
-------

===============  =======  ===================================================
``vision_pred``  D        ``[num_vision_noisy_tokens, patch_latent_dim]``
``sound_pred``   D        ``[num_sound_noisy_tokens, sound_dim]``  (gated)
``action_pred``  D        ``[num_action_noisy_tokens, action_dim]`` (gated)
===============  =======  ===================================================

Notes:
-----
* ``*_timesteps`` are **float32 regardless of the model dtype**: upstream keeps
  ``time_embedder`` in fp32 (``_keep_in_fp32_modules``) and the sinusoidal
  projection would lose integer resolution in bf16.  The embedding is cast to
  the model dtype only where it is added to the token stream.
* Optional heads are gated at *graph construction* time by the config, never
  at run time.  A ``sound_gen`` model that has no sound content in a given
  step passes **zero-length** sound tensors (``num_sound_tokens = 0``); the
  scatters and gathers then degenerate to no-ops, matching upstream's
  ``numel() > 0`` guard.  A configured head is never silently dropped.
* The packed vision predictions are returned **before** unpatchify: the host
  reshapes ``[num_vision_noisy_tokens, patch_latent_dim]`` back into
  ``[C, T, H, W]`` using the same per-item ``(T, H, W)`` shapes it used to
  patchify.  Expressing that reshape in the graph would require host-resident
  per-item shape lists, so it is a documented, precise boundary rather than a
  faked graph op.
"""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import GraphBuilder, nn

from mobius._configs._cosmos3_omni_generator import Cosmos3OmniGeneratorConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model

#: Inputs that every Cosmos3-Omni generator graph carries.
_CORE_INPUT_NAMES: tuple[str, ...] = (
    "input_ids",
    "text_indexes",
    "position_ids",
    "und_len",
    "vision_tokens",
    "vision_sequence_indexes",
    "vision_timesteps",
    "vision_timestep_token_indexes",
    "vision_mse_loss_indexes",
)

#: Additional inputs when ``config.sound_gen``.
_SOUND_INPUT_NAMES: tuple[str, ...] = (
    "sound_tokens",
    "sound_sequence_indexes",
    "sound_timesteps",
    "sound_timestep_token_indexes",
    "sound_mse_loss_indexes",
)

#: Additional inputs when ``config.action_gen``.
_ACTION_INPUT_NAMES: tuple[str, ...] = (
    "action_tokens",
    "action_domain_ids",
    "action_sequence_indexes",
    "action_timesteps",
    "action_timestep_token_indexes",
    "action_mse_loss_indexes",
    "action_pred_domain_ids",
)


def expected_input_names(config: Cosmos3OmniGeneratorConfig) -> tuple[str, ...]:
    """Return the graph input names for *config*, in declaration order."""
    names = list(_CORE_INPUT_NAMES)
    if config.sound_gen:
        names.extend(_SOUND_INPUT_NAMES)
    if config.action_gen:
        names.extend(_ACTION_INPUT_NAMES)
    return tuple(names)


def expected_output_names(config: Cosmos3OmniGeneratorConfig) -> tuple[str, ...]:
    """Return the graph output names for *config*, in declaration order."""
    names = ["vision_pred"]
    if config.sound_gen:
        names.append("sound_pred")
    if config.action_gen:
        names.append("action_pred")
    return tuple(names)


class Cosmos3OmniGeneratorTask(ModelTask):
    """Build the unified Cosmos3-Omni MoT transformer graph.

    Produces a single ``"model"`` entry.  The optimization role is
    ``"encoder"``: the graph has no KV cache, so decoder-oriented fusions
    (GroupQueryAttention with past/present) must not run on it.
    """

    model_roles: ClassVar[dict[str, str]] = {"model": "encoder"}

    def build(
        self,
        module: nn.Module,
        config: Cosmos3OmniGeneratorConfig,
    ) -> ModelPackage:
        """Wire *module* into the packed denoising-step graph.

        Args:
            module: A :class:`~mobius.models.cosmos3_omni_generator.Cosmos3OmniGeneratorModel`
                (or any module with a compatible ``forward``).
            config: Cosmos3-Omni generator configuration.  ``sound_gen`` and
                ``action_gen`` gate the optional inputs/outputs.

        Returns:
            A :class:`ModelPackage` with a single ``"model"`` entry.

        Raises:
            TypeError: If *module* does not return the
                ``(vision_pred, sound_pred, action_pred)`` contract.
        """
        config.validate()

        graph, builder = _make_graph(name="cosmos3_omni_generator")
        kwargs = self._declare_core_inputs(builder, config)
        if config.sound_gen:
            kwargs.update(self._declare_sound_inputs(builder, config))
        if config.action_gen:
            kwargs.update(self._declare_action_inputs(builder, config))

        outputs = module(builder.op, **kwargs)
        if not isinstance(outputs, tuple) or len(outputs) != 3:
            raise TypeError(
                f"{type(module).__name__} must return (vision_pred, sound_pred, action_pred)"
            )
        vision_pred, sound_pred, action_pred = outputs

        builder.add_output(vision_pred, "vision_pred")
        if config.sound_gen:
            if sound_pred is None:
                raise TypeError(
                    f"{type(module).__name__} returned sound_pred=None although "
                    "config.sound_gen is True"
                )
            builder.add_output(sound_pred, "sound_pred")
        if config.action_gen:
            if action_pred is None:
                raise TypeError(
                    f"{type(module).__name__} returned action_pred=None although "
                    "config.action_gen is True"
                )
            builder.add_output(action_pred, "action_pred")

        return ModelPackage({"model": _make_model(graph)}, config=config)

    @staticmethod
    def _declare_core_inputs(
        builder: GraphBuilder, config: Cosmos3OmniGeneratorConfig
    ) -> dict[str, ir.Value]:
        """Declare the always-present inputs (text + joint layout + vision)."""
        sequence_length = ir.SymbolicDim("sequence_length")
        num_text_tokens = ir.SymbolicDim("num_text_tokens")
        num_vision_tokens = ir.SymbolicDim("num_vision_tokens")
        num_vision_noisy = ir.SymbolicDim("num_vision_noisy_tokens")

        return {
            "input_ids": builder.input(
                "input_ids", dtype=ir.DataType.INT64, shape=[num_text_tokens]
            ),
            "text_indexes": builder.input(
                "text_indexes", dtype=ir.DataType.INT64, shape=[num_text_tokens]
            ),
            # 3-axis mRoPE positions; dim 1 also defines sequence_length.
            "position_ids": builder.input(
                "position_ids", dtype=ir.DataType.INT64, shape=[3, sequence_length]
            ),
            # Boundary between the causal understanding expert and the
            # non-causal generation expert within the joint sequence.
            "und_len": builder.input("und_len", dtype=ir.DataType.INT64, shape=[1]),
            "vision_tokens": builder.input(
                "vision_tokens",
                dtype=config.dtype,
                shape=[num_vision_tokens, config.patch_latent_dim],
            ),
            "vision_sequence_indexes": builder.input(
                "vision_sequence_indexes", dtype=ir.DataType.INT64, shape=[num_vision_tokens]
            ),
            # Timesteps stay fp32 — see the module docstring.
            "vision_timesteps": builder.input(
                "vision_timesteps", dtype=ir.DataType.FLOAT, shape=[num_vision_noisy]
            ),
            "vision_timestep_token_indexes": builder.input(
                "vision_timestep_token_indexes",
                dtype=ir.DataType.INT64,
                shape=[num_vision_noisy],
            ),
            "vision_mse_loss_indexes": builder.input(
                "vision_mse_loss_indexes", dtype=ir.DataType.INT64, shape=[num_vision_noisy]
            ),
        }

    @staticmethod
    def _declare_sound_inputs(
        builder: GraphBuilder, config: Cosmos3OmniGeneratorConfig
    ) -> dict[str, ir.Value]:
        """Declare the Sound-head inputs (only when ``config.sound_gen``)."""
        num_sound_tokens = ir.SymbolicDim("num_sound_tokens")
        num_sound_noisy = ir.SymbolicDim("num_sound_noisy_tokens")
        return {
            "sound_tokens": builder.input(
                "sound_tokens",
                dtype=config.dtype,
                shape=[num_sound_tokens, config.sound_dim],
            ),
            "sound_sequence_indexes": builder.input(
                "sound_sequence_indexes", dtype=ir.DataType.INT64, shape=[num_sound_tokens]
            ),
            "sound_timesteps": builder.input(
                "sound_timesteps", dtype=ir.DataType.FLOAT, shape=[num_sound_noisy]
            ),
            "sound_timestep_token_indexes": builder.input(
                "sound_timestep_token_indexes",
                dtype=ir.DataType.INT64,
                shape=[num_sound_noisy],
            ),
            "sound_mse_loss_indexes": builder.input(
                "sound_mse_loss_indexes", dtype=ir.DataType.INT64, shape=[num_sound_noisy]
            ),
        }

    @staticmethod
    def _declare_action_inputs(
        builder: GraphBuilder, config: Cosmos3OmniGeneratorConfig
    ) -> dict[str, ir.Value]:
        """Declare the Action-head inputs (only when ``config.action_gen``)."""
        num_action_tokens = ir.SymbolicDim("num_action_tokens")
        num_action_noisy = ir.SymbolicDim("num_action_noisy_tokens")
        return {
            "action_tokens": builder.input(
                "action_tokens",
                dtype=config.dtype,
                shape=[num_action_tokens, config.action_dim],
            ),
            # Embodiment domain selects the DomainAwareLinear weight per token.
            "action_domain_ids": builder.input(
                "action_domain_ids", dtype=ir.DataType.INT64, shape=[num_action_tokens]
            ),
            "action_sequence_indexes": builder.input(
                "action_sequence_indexes", dtype=ir.DataType.INT64, shape=[num_action_tokens]
            ),
            "action_timesteps": builder.input(
                "action_timesteps", dtype=ir.DataType.FLOAT, shape=[num_action_noisy]
            ),
            "action_timestep_token_indexes": builder.input(
                "action_timestep_token_indexes",
                dtype=ir.DataType.INT64,
                shape=[num_action_noisy],
            ),
            "action_mse_loss_indexes": builder.input(
                "action_mse_loss_indexes", dtype=ir.DataType.INT64, shape=[num_action_noisy]
            ),
            "action_pred_domain_ids": builder.input(
                "action_pred_domain_ids", dtype=ir.DataType.INT64, shape=[num_action_noisy]
            ),
        }


__all__ = [
    "Cosmos3OmniGeneratorTask",
    "expected_input_names",
    "expected_output_names",
]
