# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Hugging Face Parakeet FastConformer model for CTC speech recognition."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._configs import ParakeetCTCConfig
from mobius.components import Conv1d, ParakeetFastConformerEncoder


class ParakeetForCTCModel(nn.Module):
    """Replicate ``transformers.ParakeetForCTC`` as a single ONNX graph.

    Inputs are normalized log-mel features and their valid-frame mask. The
    FastConformer encoder downsamples time by 8 and the CTC head returns
    per-frame vocabulary logits.
    """

    default_task = "feature-ctc-asr"
    category = "Speech-to-Text"
    config_class = ParakeetCTCConfig

    def __init__(self, config: ParakeetCTCConfig):
        super().__init__()
        if config.dtype == ir.DataType.BFLOAT16:
            raise ValueError(
                "Parakeet CTC bf16 is disabled because ONNX Runtime produces "
                "incorrect CTC logits; use dtype='f16' or dtype='f32'."
            )
        self.encoder = ParakeetFastConformerEncoder(config)
        self.ctc_head = Conv1d(config.hidden_size, config.vocab_size, kernel_size=1)

    def forward(
        self,
        op: OpBuilder,
        input_features: ir.Value,
        attention_mask: ir.Value,
    ) -> ir.Value:
        hidden_states, _ = self.encoder(op, input_features, attention_mask)
        # The checkpoint stores the CTC projection as Conv1d (vocab, hidden, 1).
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
        logits = self.ctc_head(op, hidden_states)
        return op.Transpose(logits, perm=[0, 2, 1])

    def preprocess_weights(self, state_dict: dict[str, object]) -> dict[str, object]:
        """Drop PyTorch-only BatchNorm counters; all tensor names align directly."""
        return {
            name: value
            for name, value in state_dict.items()
            if not name.endswith(".num_batches_tracked")
        }
