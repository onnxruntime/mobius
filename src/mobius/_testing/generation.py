# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Autoregressive text generation using ONNX Runtime.

Provides self-contained generation loops with KV cache management,
without depending on onnxruntime-genai.  Supports causal-LM (decoder-
only) and seq2seq (encoder-decoder) architectures.
"""

from __future__ import annotations

import numpy as np
import torch

from mobius._configs import ArchitectureConfig
from mobius._testing.ort_inference import OnnxModelSession


class OnnxGenerator:
    """Greedy autoregressive text generator backed by an ONNX model.

    Manages the KV cache, attention mask growth, and position ID
    bookkeeping for the autoregressive generation loop.

    Example::

        session = OnnxModelSession(model)
        gen = OnnxGenerator(session, config)
        output_ids = gen.generate(input_ids, max_new_tokens=20)
    """

    def __init__(
        self,
        session: OnnxModelSession,
        config: ArchitectureConfig,
    ):
        self.session = session
        self.config = config

    def generate(
        self,
        input_ids: np.ndarray,
        max_new_tokens: int = 20,
        eos_token_id: int | None = None,
    ) -> np.ndarray:
        """Generate tokens autoregressively using greedy decoding.

        Args:
            input_ids: [batch, seq_len] int64 prompt token IDs.
            max_new_tokens: Maximum number of new tokens to generate.
            eos_token_id: If set, stop generation when this token is produced.

        Returns:
            [batch, seq_len + generated_len] int64 array of all token IDs
            (prompt + generated).
        """
        batch_size, _prompt_len = input_ids.shape
        num_layers = self.config.num_hidden_layers
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim
        layer_types = self.config.layer_types or []

        # Initialize empty past KV / recurrent state per layer
        past_kv: dict[str, np.ndarray] = {}
        for i in range(num_layers):
            ltype = layer_types[i] if i < len(layer_types) else "full_attention"
            if ltype in ("mamba", "mamba2", "linear_attention"):
                # Recurrent states: use shapes declared by the ONNX model
                if ltype == "linear_attention":
                    suffixes = ("conv_state", "recurrent_state")
                else:
                    suffixes = ("conv_state", "ssm_state")
                for suffix in suffixes:
                    name = f"past_key_values.{i}.{suffix}"
                    shape = self.session.get_input_shape(name) or []
                    static = [d if isinstance(d, int) and d > 0 else batch_size for d in shape]
                    past_kv[name] = np.zeros(static, dtype=np.float32)
            elif ltype == "lightning_attention":
                # Single recurrent state only (no conv_state)
                name = f"past_key_values.{i}.recurrent_state"
                shape = self.session.get_input_shape(name) or []
                static = [d if isinstance(d, int) and d > 0 else batch_size for d in shape]
                past_kv[name] = np.zeros(static, dtype=np.float32)
            elif ltype == "conv":
                # ShortConv conv_state only (no SSM state)
                name = f"past_key_values.{i}.conv_state"
                shape = self.session.get_input_shape(name) or []
                static = [d if isinstance(d, int) and d > 0 else batch_size for d in shape]
                past_kv[name] = np.zeros(static, dtype=np.float32)
            else:
                past_kv[f"past_key_values.{i}.key"] = np.zeros(
                    (batch_size, num_kv_heads, 0, head_dim), dtype=np.float32
                )
                past_kv[f"past_key_values.{i}.value"] = np.zeros(
                    (batch_size, num_kv_heads, 0, head_dim), dtype=np.float32
                )

        all_ids = input_ids.copy()

        # First step: process the full prompt
        cur_input_ids = input_ids
        past_seq_len = 0

        for _step in range(max_new_tokens):
            cur_seq_len = cur_input_ids.shape[1]
            total_seq_len = past_seq_len + cur_seq_len

            attention_mask = np.ones((batch_size, total_seq_len), dtype=np.int64)
            position_ids = np.arange(past_seq_len, total_seq_len, dtype=np.int64)[
                np.newaxis, :
            ].repeat(batch_size, axis=0)

            feeds = {
                "input_ids": cur_input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                **past_kv,
            }

            outputs = self.session.run(feeds)

            # Extract logits and take argmax of last token
            logits = outputs["logits"]  # [batch, cur_seq_len, vocab]
            next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            # next_token: [batch, 1]

            all_ids = np.concatenate([all_ids, next_token], axis=1)

            # Check EOS
            if eos_token_id is not None and np.all(next_token == eos_token_id):
                break

            # Update past KV / recurrent state from present outputs
            for i in range(num_layers):
                ltype = layer_types[i] if i < len(layer_types) else "full_attention"
                if ltype in ("mamba", "mamba2", "linear_attention"):
                    if ltype == "linear_attention":
                        suffixes = ("conv_state", "recurrent_state")
                    else:
                        suffixes = ("conv_state", "ssm_state")
                    for suffix in suffixes:
                        src = f"present.{i}.{suffix}"
                        dst = f"past_key_values.{i}.{suffix}"
                        if src in outputs:
                            past_kv[dst] = outputs[src]
                elif ltype == "lightning_attention":
                    src = f"present.{i}.recurrent_state"
                    dst = f"past_key_values.{i}.recurrent_state"
                    if src in outputs:
                        past_kv[dst] = outputs[src]
                elif ltype == "conv":
                    src = f"present.{i}.conv_state"
                    dst = f"past_key_values.{i}.conv_state"
                    if src in outputs:
                        past_kv[dst] = outputs[src]
                else:
                    past_kv[f"past_key_values.{i}.key"] = outputs[f"present.{i}.key"]
                    past_kv[f"past_key_values.{i}.value"] = outputs[f"present.{i}.value"]

            # Next step: only the new token
            cur_input_ids = next_token.astype(np.int64)
            past_seq_len = total_seq_len

        return all_ids


class OnnxSeq2SeqGenerator:
    """Greedy generation for encoder-decoder (seq2seq) ONNX models.

    Runs the encoder once, then autoregressively decodes using the
    decoder with cross-attention to encoder hidden states.  Manages
    both self-attention and cross-attention KV caches.

    Note: This generator is designed for text-to-text seq2seq models
    (BART, T5, mBART, etc.) where both encoder and decoder use
    ``input_ids``.  It is NOT suitable for speech-to-text models
    (e.g. Whisper) which require ``input_features`` for the encoder
    and ``decoder_input_ids`` + ``position_ids`` for the decoder.

    Example::

        enc_session = OnnxModelSession(pkg["encoder"])
        dec_session = OnnxModelSession(pkg["decoder"])
        gen = OnnxSeq2SeqGenerator(enc_session, dec_session, config)
        output_ids = gen.generate(input_ids, max_new_tokens=20)
    """

    def __init__(
        self,
        enc_session: OnnxModelSession,
        dec_session: OnnxModelSession,
        config: ArchitectureConfig,
    ):
        self.enc_session = enc_session
        self.dec_session = dec_session
        self.config = config

    def generate(
        self,
        input_ids: np.ndarray,
        max_new_tokens: int = 20,
        eos_token_id: int | None = None,
        decoder_start_token_id: int = 0,
    ) -> np.ndarray:
        """Generate tokens from encoder input using greedy decoding.

        Args:
            input_ids: [batch, src_seq_len] int64 encoder input tokens.
            max_new_tokens: Maximum number of tokens to generate.
            eos_token_id: If set, stop when this token is produced.
            decoder_start_token_id: Token to seed the decoder.

        Returns:
            [batch, generated_len] int64 array of generated token IDs
            (decoder start token + generated, no encoder input).
        """
        batch_size = input_ids.shape[0]
        src_seq_len = input_ids.shape[1]

        # Step 1: Run encoder once
        enc_feeds = {
            "input_ids": input_ids,
            "attention_mask": np.ones_like(input_ids),
        }
        enc_outputs = self.enc_session.run(enc_feeds)

        # Extract encoder hidden states
        enc_hidden = None
        for key in ("encoder_hidden_states", "last_hidden_state"):
            if key in enc_outputs:
                enc_hidden = enc_outputs[key]
                break
        if enc_hidden is None:
            raise KeyError(
                f"Encoder output missing hidden states. Keys: {sorted(enc_outputs.keys())}"
            )

        # Step 2: Initialize decoder KV caches
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim
        past_kv: dict[str, np.ndarray] = {}

        for name in self.dec_session.input_names:
            if not name.startswith("past_key_values."):
                continue
            if ".cross." in name:
                # Cross-attention cache: starts empty, populated on first step
                past_kv[name] = np.zeros(
                    (batch_size, num_kv_heads, 0, head_dim),
                    dtype=np.float32,
                )
            else:
                # Self-attention cache: grows each step
                past_kv[name] = np.zeros(
                    (batch_size, num_kv_heads, 0, head_dim),
                    dtype=np.float32,
                )

        # Step 3: Autoregressive decode loop
        cur_dec_ids = np.full((batch_size, 1), decoder_start_token_id, dtype=np.int64)
        all_ids = cur_dec_ids.copy()

        for _step in range(max_new_tokens):
            dec_feeds: dict[str, np.ndarray] = {
                "input_ids": cur_dec_ids,
                "encoder_hidden_states": enc_hidden,
                "attention_mask": np.ones((batch_size, src_seq_len), dtype=np.int64),
                **past_kv,
            }

            outputs = self.dec_session.run(dec_feeds)

            # Extract logits and take argmax of last token
            logits = outputs["logits"]  # [batch, dec_seq_len, vocab]
            next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True)

            all_ids = np.concatenate([all_ids, next_token], axis=1)

            # Check EOS
            if eos_token_id is not None and np.all(next_token == eos_token_id):
                break

            # Update KV caches from present outputs
            for name in list(past_kv.keys()):
                # past_key_values.N.self.key → present.N.self.key
                layer_suffix = name.replace("past_key_values.", "")
                present_name = f"present.{layer_suffix}"
                if present_name in outputs:
                    past_kv[name] = outputs[present_name]

            # Next step: only the new token
            cur_dec_ids = next_token.astype(np.int64)

        return all_ids


def torch_generate_greedy(
    model,
    input_ids: np.ndarray,
    max_new_tokens: int = 20,
    eos_token_id: int | None = None,
) -> np.ndarray:
    """Greedy generation using a HuggingFace model (for reference comparison).

    Uses the same greedy argmax logic as OnnxGenerator so results are
    directly comparable (no sampling, no temperature).

    Args:
        model: HuggingFace causal LM model in eval mode.
        input_ids: [batch, seq_len] int64 numpy array.
        max_new_tokens: Maximum new tokens.
        eos_token_id: Stop token.

    Returns:
        [batch, total_len] int64 numpy array.
    """
    device = next(model.parameters()).device
    ids = torch.from_numpy(input_ids).to(device)
    attention_mask = torch.ones_like(ids)

    with torch.no_grad():
        output = model.generate(
            ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_token_id,
        )

    return output.cpu().numpy()
