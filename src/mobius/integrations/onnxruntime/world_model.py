# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Stateful ONNX Runtime execution for the world-model task contract."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import ml_dtypes
import numpy as np
from numpy.typing import ArrayLike

from mobius.tasks._world_model import WorldModelTask


class _NodeMetadata(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def shape(self) -> Sequence[int | str | None]: ...

    @property
    def type(self) -> str: ...


class WorldModelSession(Protocol):
    """Minimal inference-session interface required by :class:`WorldModelRunner`."""

    def get_inputs(self) -> Sequence[_NodeMetadata]: ...

    def get_outputs(self) -> Sequence[_NodeMetadata]: ...

    def run(
        self,
        output_names: Sequence[str],
        input_feed: dict[str, np.ndarray],
    ) -> Sequence[Any]: ...


@dataclass(frozen=True)
class WorldModelStepOutput:
    """Outputs from one world-model transition."""

    next_state: np.ndarray
    observation_prediction: np.ndarray
    reward: np.ndarray
    continuation: np.ndarray


_ORT_TYPE_TO_NUMPY: dict[str, np.dtype] = {
    "tensor(float)": np.dtype(np.float32),
    "tensor(float16)": np.dtype(np.float16),
    "tensor(double)": np.dtype(np.float64),
    "tensor(bfloat16)": np.dtype(ml_dtypes.bfloat16),
    "tensor(int64)": np.dtype(np.int64),
    "tensor(int32)": np.dtype(np.int32),
    "tensor(int16)": np.dtype(np.int16),
    "tensor(int8)": np.dtype(np.int8),
    "tensor(uint64)": np.dtype(np.uint64),
    "tensor(uint32)": np.dtype(np.uint32),
    "tensor(uint16)": np.dtype(np.uint16),
    "tensor(uint8)": np.dtype(np.uint8),
    "tensor(bool)": np.dtype(np.bool_),
}


class WorldModelRunner:
    """Run a Mobius world-model graph while preserving recurrent state."""

    def __init__(self, session: WorldModelSession):
        self._session = session
        self._inputs = {node.name: node for node in session.get_inputs()}
        self._outputs = {node.name: node for node in session.get_outputs()}
        self._validate_contract()
        self._state: np.ndarray | None = None

    @classmethod
    def from_path(
        cls,
        model_path: str | os.PathLike[str],
        *,
        providers: Sequence[str] | None = None,
        session_options: object | None = None,
    ) -> WorldModelRunner:
        """Create a runner from an ONNX file without making ORT a core dependency."""
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            if exc.name != "onnxruntime":
                raise
            raise ImportError(
                "WorldModelRunner.from_path() requires onnxruntime; "
                "install mobius-onnx[runtime]"
            ) from exc

        kwargs: dict[str, object] = {}
        if providers is not None:
            kwargs["providers"] = list(providers)
        if session_options is not None:
            kwargs["sess_options"] = session_options
        return cls(ort.InferenceSession(os.fspath(model_path), **kwargs))

    @property
    def session(self) -> WorldModelSession:
        """Underlying inference session."""
        return self._session

    @property
    def state(self) -> np.ndarray | None:
        """Recurrent state used by the next step."""
        return self._state

    def reset(
        self,
        state: ArrayLike | None = None,
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Reset recurrent state to an explicit value or a zero tensor."""
        if state is not None:
            prepared = self._prepare_input("state", state)
            if batch_size is not None and prepared.shape[0] != batch_size:
                raise ValueError(
                    f"state batch dimension is {prepared.shape[0]}, expected {batch_size}"
                )
        else:
            prepared = self._make_zero_state(batch_size or 1)
        self._state = prepared
        return prepared

    def step(
        self,
        observation: ArrayLike,
        action: ArrayLike,
        *,
        state: ArrayLike | None = None,
    ) -> WorldModelStepOutput:
        """Execute one transition and retain its ``next_state``."""
        observation_array = self._prepare_input("observation", observation)
        action_array = self._prepare_input("action", action)
        batch_size = observation_array.shape[0]
        if action_array.shape[0] != batch_size:
            raise ValueError(
                "observation and action batch dimensions differ: "
                f"{batch_size} != {action_array.shape[0]}"
            )

        if state is not None:
            current_state = self._prepare_input("state", state)
        elif self._state is not None:
            current_state = self._state
        else:
            current_state = self._make_zero_state(batch_size)
        if current_state.shape[0] != batch_size:
            raise ValueError(
                "observation and state batch dimensions differ: "
                f"{batch_size} != {current_state.shape[0]}"
            )

        values = self._session.run(
            list(WorldModelTask.output_names),
            {
                "observation": observation_array,
                "action": action_array,
                "state": current_state,
            },
        )
        if len(values) != len(WorldModelTask.output_names):
            raise RuntimeError(
                f"session returned {len(values)} outputs, "
                f"expected {len(WorldModelTask.output_names)}"
            )

        arrays = {
            name: self._validate_output(name, value)
            for name, value in zip(WorldModelTask.output_names, values, strict=True)
        }
        if arrays["next_state"].shape != current_state.shape:
            raise ValueError(
                "next_state shape is incompatible with recurrent state: "
                f"{arrays['next_state'].shape} != {current_state.shape}"
            )
        if arrays["observation_prediction"].shape != observation_array.shape:
            raise ValueError(
                "observation_prediction shape is incompatible with observation: "
                f"{arrays['observation_prediction'].shape} != {observation_array.shape}"
            )
        expected_scalar_shape = (batch_size, 1)
        for name in ("reward", "continuation"):
            if arrays[name].shape != expected_scalar_shape:
                raise ValueError(
                    f"{name} has shape {arrays[name].shape}, expected {expected_scalar_shape}"
                )
        output = WorldModelStepOutput(
            next_state=arrays["next_state"],
            observation_prediction=arrays["observation_prediction"],
            reward=arrays["reward"],
            continuation=arrays["continuation"],
        )
        self._state = output.next_state
        return output

    def rollout(
        self,
        observations: Sequence[ArrayLike],
        actions: Sequence[ArrayLike],
        *,
        initial_state: ArrayLike | None = None,
    ) -> tuple[WorldModelStepOutput, ...]:
        """Run a sequence of transitions along the leading time dimension."""
        if len(observations) != len(actions):
            raise ValueError("observations and actions must contain the same number of steps")
        if len(observations) == 0:
            raise ValueError("rollout requires at least one step")
        if initial_state is not None:
            self.reset(initial_state)
        else:
            first_observation = self._prepare_input("observation", observations[0])
            self.reset(batch_size=first_observation.shape[0])

        return tuple(
            self.step(observation, action)
            for observation, action in zip(observations, actions, strict=True)
        )

    def _validate_contract(self) -> None:
        expected_inputs = set(WorldModelTask.input_names)
        expected_outputs = set(WorldModelTask.output_names)
        actual_inputs = set(self._inputs)
        actual_outputs = set(self._outputs)
        if actual_inputs != expected_inputs:
            raise ValueError(
                "world-model input contract mismatch: "
                f"expected {sorted(expected_inputs)}, got {sorted(actual_inputs)}"
            )
        if actual_outputs != expected_outputs:
            raise ValueError(
                "world-model output contract mismatch: "
                f"expected {sorted(expected_outputs)}, got {sorted(actual_outputs)}"
            )
        state_input = self._inputs["state"]
        next_state_output = self._outputs["next_state"]
        if state_input.type != next_state_output.type:
            raise ValueError(
                "next_state dtype must match state dtype: "
                f"{next_state_output.type!r} != {state_input.type!r}"
            )
        if not self._shapes_are_recurrently_compatible(
            state_input.shape,
            next_state_output.shape,
        ):
            raise ValueError(
                "next_state shape must be compatible with state shape: "
                f"{tuple(next_state_output.shape)} != {tuple(state_input.shape)}"
            )

    def _prepare_input(self, name: str, value: ArrayLike) -> np.ndarray:
        node = self._inputs[name]
        array = np.ascontiguousarray(value, dtype=self._numpy_dtype(node))
        self._validate_shape(name, array, node)
        return array

    def _validate_output(self, name: str, value: Any) -> np.ndarray:
        node = self._outputs[name]
        array = np.asarray(value)
        expected_dtype = self._numpy_dtype(node)
        if array.dtype != expected_dtype:
            raise TypeError(f"{name} has dtype {array.dtype}, expected {expected_dtype}")
        self._validate_shape(name, array, node)
        return array

    def _make_zero_state(self, batch_size: int) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        node = self._inputs["state"]
        shape: list[int] = []
        for axis, dim in enumerate(node.shape):
            if axis == 0:
                if isinstance(dim, int) and dim != batch_size:
                    raise ValueError(
                        f"state model batch dimension is fixed at {dim}, got {batch_size}"
                    )
                shape.append(batch_size)
            elif isinstance(dim, int):
                shape.append(dim)
            else:
                raise ValueError(
                    "cannot create zero state because a non-batch state dimension "
                    f"is dynamic ({dim!r}); pass an explicit state to reset() or step()"
                )
        return np.zeros(shape, dtype=self._numpy_dtype(node))

    @staticmethod
    def _numpy_dtype(node: _NodeMetadata) -> np.dtype:
        try:
            return _ORT_TYPE_TO_NUMPY[node.type]
        except KeyError as exc:
            raise TypeError(f"unsupported ONNX Runtime tensor type {node.type!r}") from exc

    @staticmethod
    def _validate_shape(name: str, array: np.ndarray, node: _NodeMetadata) -> None:
        expected = tuple(node.shape)
        if array.ndim != len(expected):
            raise ValueError(
                f"{name} has rank {array.ndim}, expected rank {len(expected)} "
                f"with shape {expected}"
            )
        for axis, (actual_dim, expected_dim) in enumerate(
            zip(array.shape, expected, strict=True)
        ):
            if isinstance(expected_dim, int) and actual_dim != expected_dim:
                raise ValueError(
                    f"{name} dimension {axis} is {actual_dim}, expected {expected_dim}"
                )

    @staticmethod
    def _shapes_are_recurrently_compatible(
        state_shape: Sequence[int | str | None],
        next_state_shape: Sequence[int | str | None],
    ) -> bool:
        if len(state_shape) != len(next_state_shape):
            return False
        return all(
            state_dim == next_state_dim
            if isinstance(state_dim, int) and isinstance(next_state_dim, int)
            else True
            for state_dim, next_state_dim in zip(
                state_shape,
                next_state_shape,
                strict=True,
            )
        )
