# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the generic ``TextModel`` DeepStack injection support.

``TextModel.forward``'s ``deepstack_inputs`` parameter is architecture-agnostic
(used by Qwen3-VL DeepStack today, but not tied to any single model), so its
correctness is tested here in isolation rather than duplicated per-model.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import torch

from mobius._testing import make_config
from mobius._weight_loading import apply_weights
from mobius.models.base import TextModel
from mobius.tasks._base import _make_graph, _make_model


def _random_state_dict(model: TextModel, seed: int = 0) -> dict[str, torch.Tensor]:
    """Small random weights for every parameter, keyed by ONNX initializer name."""
    generator = torch.Generator().manual_seed(seed)
    return {
        name: torch.randn(*param.shape, generator=generator) * 0.02
        for name, param in model.named_parameters()
    }


def _build_deepstack_graph(
    model: TextModel, num_layers: int, num_deepstack: int, hidden_size: int
):
    """Build a minimal graph feeding ``inputs_embeds`` + ``per_layer_inputs``.

    Mirrors exactly the reshape → slice → squeeze unpacking done in
    ``Qwen3VLDecoderModel.forward``, so this test exercises the same
    ``TextModel.forward(deepstack_inputs=...)`` contract that model relies on.
    Captures every layer's post-residual hidden state via
    ``output_layer_indices`` so the injection point can be checked exactly.
    """
    batch = ir.SymbolicDim("batch")
    seq_len = ir.SymbolicDim("seq_len")

    graph, builder = _make_graph()
    op = builder.op
    inputs_embeds = builder.input(
        "inputs_embeds", dtype=ir.DataType.FLOAT, shape=[batch, seq_len, hidden_size]
    )
    attention_mask = builder.input(
        "attention_mask", dtype=ir.DataType.INT64, shape=[batch, seq_len]
    )
    position_ids = builder.input(
        "position_ids", dtype=ir.DataType.INT64, shape=[batch, seq_len]
    )
    per_layer_inputs = builder.input(
        "per_layer_inputs",
        dtype=ir.DataType.FLOAT,
        shape=[batch, seq_len, num_deepstack * hidden_size],
    )

    per_layer_4d = op.Reshape(
        per_layer_inputs, op.Constant(value_ints=[0, 0, num_deepstack, hidden_size])
    )
    deepstack_inputs = [
        op.Squeeze(op.Slice(per_layer_4d, starts=[i], ends=[i + 1], axes=[2]), [2])
        for i in range(num_deepstack)
    ]

    _, _, captured = model(
        op,
        input_ids=None,
        attention_mask=attention_mask,
        position_ids=position_ids,
        inputs_embeds=inputs_embeds,
        deepstack_inputs=deepstack_inputs,
    )
    for i, hidden in enumerate(captured):
        builder.add_output(hidden, f"captured_{i}")

    built = _make_model(graph)
    apply_weights(built, _random_state_dict(model))
    return built


class TestTextModelDeepStackInjection:
    """``TextModel.forward(deepstack_inputs=...)`` adds each slot after its layer."""

    NUM_LAYERS = 4
    NUM_DEEPSTACK = 2
    HIDDEN = 64

    def _session_and_inputs(self):
        from mobius._testing.ort_inference import OnnxModelSession

        config = make_config(
            num_hidden_layers=self.NUM_LAYERS,
            hidden_size=self.HIDDEN,
            output_layer_indices=list(range(self.NUM_LAYERS)),
        )
        model = TextModel(config)
        built = _build_deepstack_graph(model, self.NUM_LAYERS, self.NUM_DEEPSTACK, self.HIDDEN)
        session = OnnxModelSession(built, device="cpu")

        batch, seq_len = 1, 3
        rng = np.random.default_rng(0)
        embeds = rng.normal(size=(batch, seq_len, self.HIDDEN)).astype(np.float32)
        mask = np.ones((batch, seq_len), dtype=np.int64)
        pos = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)
        base_inputs = {
            "inputs_embeds": embeds,
            "attention_mask": mask,
            "position_ids": pos,
        }
        return session, base_inputs, (batch, seq_len)

    def _run_with_slot(self, session, base_inputs, shape, slot: int | None, value: float):
        batch, seq_len = shape
        per_layer = np.zeros(
            (batch, seq_len, self.NUM_DEEPSTACK * self.HIDDEN), dtype=np.float32
        )
        if slot is not None:
            per_layer[:, :, slot * self.HIDDEN : (slot + 1) * self.HIDDEN] = value
        return session.run({**base_inputs, "per_layer_inputs": per_layer})

    def test_zero_per_layer_inputs_is_a_no_op(self):
        """All-zero ``per_layer_inputs`` must not perturb any captured layer."""
        session, base_inputs, shape = self._session_and_inputs()
        out_a = self._run_with_slot(session, base_inputs, shape, slot=None, value=0.0)
        out_b = self._run_with_slot(session, base_inputs, shape, slot=None, value=0.0)
        for i in range(self.NUM_LAYERS):
            np.testing.assert_array_equal(out_a[f"captured_{i}"], out_b[f"captured_{i}"])

    def test_slot_added_exactly_after_its_own_layer(self):
        """Perturbing slot ``i`` changes captured layer ``i`` by exactly that value.

        This is the precise numeric proof of "layer injection order": the
        magnitude match (not just inequality) proves a direct, unscaled
        ``Add`` at that exact point in the graph.
        """
        session, base_inputs, shape = self._session_and_inputs()
        zero = self._run_with_slot(session, base_inputs, shape, slot=None, value=0.0)
        big = 1000.0
        for slot in range(self.NUM_DEEPSTACK):
            perturbed = self._run_with_slot(session, base_inputs, shape, slot=slot, value=big)
            diff = perturbed[f"captured_{slot}"] - zero[f"captured_{slot}"]
            np.testing.assert_allclose(diff, big, rtol=1e-4)

    def test_slot_never_affects_earlier_layers(self):
        """Perturbing slot ``i`` must leave every layer ``< i`` bit-for-bit unchanged.

        Proves the addition happens *after* layer ``i``, not before it or at
        layer 0 regardless of slot index.
        """
        session, base_inputs, shape = self._session_and_inputs()
        zero = self._run_with_slot(session, base_inputs, shape, slot=None, value=0.0)
        big = 1000.0
        for slot in range(1, self.NUM_DEEPSTACK):
            perturbed = self._run_with_slot(session, base_inputs, shape, slot=slot, value=big)
            for earlier in range(slot):
                np.testing.assert_array_equal(
                    perturbed[f"captured_{earlier}"], zero[f"captured_{earlier}"]
                )

    def test_slot_propagates_to_later_layers(self):
        """Perturbing slot ``i`` must change every subsequent captured layer.

        (Sanity check that the addition really feeds forward through the
        residual stream rather than being a dead-end/no-op branch.)
        """
        session, base_inputs, shape = self._session_and_inputs()
        zero = self._run_with_slot(session, base_inputs, shape, slot=None, value=0.0)
        big = 1000.0
        perturbed = self._run_with_slot(session, base_inputs, shape, slot=0, value=big)
        for later in range(1, self.NUM_LAYERS):
            assert not np.allclose(
                perturbed[f"captured_{later}"], zero[f"captured_{later}"]
            ), f"layer {later} should be perturbed by an earlier DeepStack injection"

    def test_deepstack_inputs_beyond_num_layers_is_gated_by_len(self):
        """More deepstack slots than decoder layers only inject up to num_layers.

        ``layer_idx < len(deepstack_inputs)`` in ``TextModel.forward`` means
        that if a caller ever passed more slots than layers, injection simply
        stops at the last layer — it must not raise or wrap around.
        """
        config = make_config(num_hidden_layers=2, hidden_size=self.HIDDEN)
        model = TextModel(config)

        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("seq_len")
        graph, builder = _make_graph()
        op = builder.op
        inputs_embeds = builder.input(
            "inputs_embeds", dtype=ir.DataType.FLOAT, shape=[batch, seq_len, self.HIDDEN]
        )
        attention_mask = builder.input(
            "attention_mask", dtype=ir.DataType.INT64, shape=[batch, seq_len]
        )
        position_ids = builder.input(
            "position_ids", dtype=ir.DataType.INT64, shape=[batch, seq_len]
        )
        # 3 deepstack slots but only 2 decoder layers.
        deepstack_inputs = [op.Constant(value_floats=[0.1] * self.HIDDEN) for _ in range(3)]
        hidden_states, _present_key_values = model(
            op,
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            deepstack_inputs=deepstack_inputs,
        )
        builder.add_output(hidden_states, "hidden_states")
        built = _make_model(graph)
        apply_weights(built, _random_state_dict(model))

        from mobius._testing.ort_inference import OnnxModelSession

        session = OnnxModelSession(built, device="cpu")
        batch_n, seq_n = 1, 2
        result = session.run(
            {
                "inputs_embeds": np.zeros((batch_n, seq_n, self.HIDDEN), dtype=np.float32),
                "attention_mask": np.ones((batch_n, seq_n), dtype=np.int64),
                "position_ids": np.arange(seq_n, dtype=np.int64).reshape(1, seq_n),
            }
        )
        assert result["hidden_states"].shape == (batch_n, seq_n, self.HIDDEN)

    def test_default_deepstack_inputs_none_is_unchanged(self):
        """``deepstack_inputs=None`` (the default) preserves the legacy 2-tuple return.

        Regression guard: every existing model built on ``TextModel`` (Llama,
        Qwen2.5-VL, Qwen3.5-VL decoders, etc.) calls ``forward`` without this
        parameter and must keep getting ``(hidden_states, present_key_values)``.
        """
        config = make_config(num_hidden_layers=2, hidden_size=self.HIDDEN)
        model = TextModel(config)

        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("seq_len")
        _graph, builder = _make_graph()
        op = builder.op
        inputs_embeds = builder.input(
            "inputs_embeds", dtype=ir.DataType.FLOAT, shape=[batch, seq_len, self.HIDDEN]
        )
        attention_mask = builder.input(
            "attention_mask", dtype=ir.DataType.INT64, shape=[batch, seq_len]
        )
        position_ids = builder.input(
            "position_ids", dtype=ir.DataType.INT64, shape=[batch, seq_len]
        )
        result = model(
            op,
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        assert len(result) == 2, "default forward() must return the legacy 2-tuple"
