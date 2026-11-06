# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Ordinary fresh-decoder caller for explicit-phase experimental validation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

_Result = TypeVar("_Result")


def validate_phase_session(session: Any) -> dict[str, Any]:
    """Validate real runtime graph metadata; never disguise an old decoder."""
    inputs = session.get_inputs()
    matches = [value for value in inputs if value.name == "has_previous_state"]
    if len(matches) != 1 or matches[0].type != "tensor(bool)" or matches[0].shape != []:
        raise ValueError("Actual decoder must expose canonical has_previous_state BOOL []")
    return {
        "name": matches[0].name,
        "type": matches[0].type,
        "shape": matches[0].shape,
        "inputs": [
            {"name": value.name, "type": value.type, "shape": value.shape} for value in inputs
        ],
        "outputs": [
            {"name": value.name, "type": value.type, "shape": value.shape}
            for value in session.get_outputs()
        ],
    }


class ExplicitCachePhaseCaller(Generic[_Result]):
    """Own phase independently of lengths/offsets, advancing only after success.

    The supplied synchronous callback executes exactly one decoder call and
    owns cache output commitment. This validation adapter is not a scheduler,
    Engine, prompt chunker, replay/rewind helper, or CUDA Graph capture route.
    """

    def __init__(
        self,
        execute: Callable[[dict[str, NDArray[Any]]], _Result],
        *,
        route: str = "ordinary",
        prompt_chunk_size: int | None = None,
        cuda_graph_capture: bool = False,
    ) -> None:
        if route != "ordinary" or prompt_chunk_size is not None or cuda_graph_capture:
            raise ValueError(
                "Source-chunk candidate forbids scheduler/Engine, prompt chunking and capture"
            )
        self._execute = execute
        self._phase = np.asarray(False, dtype=np.bool_)
        self._failed = False

    @property
    def has_previous_state(self) -> bool:
        return bool(self._phase)

    def __call__(self, feeds: dict[str, NDArray[Any]]) -> _Result:
        if self._failed:
            raise RuntimeError("Decoder caller is poisoned after failure; no retry/reset")
        try:
            return self._call(feeds)
        except BaseException:
            self._failed = True
            raise

    def _call(self, feeds: dict[str, NDArray[Any]]) -> _Result:
        tokens = feeds.get("input_ids", feeds.get("inputs_embeds"))
        if tokens is None or tokens.ndim not in (2, 3) or tokens.shape[1] <= 0:
            raise ValueError("Decoder requires nonempty batched token IDs or embeddings")
        if self.has_previous_state and tokens.shape[1] != 1:
            raise ValueError("Source cached S>1 is unsupported")
        mask = feeds.get("attention_mask")
        if mask is not None and (
            mask.ndim != 2
            or mask.dtype not in (np.dtype(np.int64), np.dtype(np.bool_))
            or mask.shape[0] != tokens.shape[0]
            or mask.shape[1] < tokens.shape[1]
            or not np.all((mask == 0) | (mask == 1))
        ):
            raise ValueError("Source-chunk caller requires a 2D INT64/BOOL binary total mask")
        supplied = feeds.get("has_previous_state")
        if supplied is not None and (
            supplied.dtype != np.bool_
            or supplied.shape != ()
            or bool(supplied) != self.has_previous_state
        ):
            raise ValueError(
                "Provided has_previous_state contradicts owned successful cache phase"
            )
        result = self._execute({**feeds, "has_previous_state": self._phase})
        # The callback owns atomic cache-output commitment; this wrapper only
        # commits phase. Never advance on preparation or output allocation.
        self._phase[...] = True
        return result

    def rewind(self) -> None:
        self._failed = True
        raise NotImplementedError("Source-chunk candidate does not support rewind")
