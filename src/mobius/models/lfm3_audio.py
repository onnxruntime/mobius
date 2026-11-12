# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""LiquidAI LFM3 Audio spectrogram frontend."""

from __future__ import annotations

import dataclasses

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import Lfm2Config
from mobius._configs.lfm3_audio import Lfm3AudioConfig
from mobius.components import (
    Conv1d,
    Conv2d,
    Embedding,
    LayerNorm,
    Linear,
    SiLU,
    initialize_rope,
)
from mobius.models.lfm2 import (
    Lfm2DecoderLayer,
    Lfm2RMSNorm,
    apply_lfm2_config_defaults,
    rename_lfm2_weight_key,
)
from mobius.models.lfm3 import Lfm3Attention, Lfm3RMSNorm, Lfm3TextModel


class Lfm3AudioDownsampler(nn.Module):
    """Replicate HuggingFace Lfm3AudioDownsampler's masked stride-eight frontend.

    Inputs are float log-mel features (B, mel_bins, frames) and int64 valid
    frame lengths (B,). Outputs have shape (B, hidden_size, ceil(frames / 8)).
    Padding must be masked between convolutions, not only at graph entry:
    convolution biases otherwise leak padded activations into valid frames.
    """

    def __init__(self, num_mel_bins: int, conv_channels: int, hidden_size: int):
        super().__init__()
        if num_mel_bins <= 0 or num_mel_bins % 8:
            raise ValueError(
                f"LFM3 Audio num_mel_bins must be a positive multiple of 8, "
                f"got {num_mel_bins}; use the checkpoint's audio encoder configuration."
            )
        self.conv_in = Conv2d(1, conv_channels, kernel_size=3, stride=2, padding=1)
        self.conv_mid = Conv2d(
            conv_channels,
            conv_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=conv_channels,
        )
        self.lin_mid = Conv2d(conv_channels, conv_channels, kernel_size=1)
        self.conv_end = Conv2d(
            conv_channels,
            conv_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=conv_channels,
        )
        self.lin_end = Conv2d(conv_channels, conv_channels, kernel_size=1)
        self.lin_out = Conv1d(num_mel_bins * conv_channels // 8, hidden_size, kernel_size=1)
        self.act = SiLU()

    @staticmethod
    def _mask_padding(op: OpBuilder, hidden_states: ir.Value, lengths: ir.Value) -> ir.Value:
        frames = op.Squeeze(op.Shape(hidden_states, start=3, end=4), [0])
        indices = op.Range(0, frames, 1)
        valid = op.Less(op.Unsqueeze(indices, [0]), op.Unsqueeze(lengths, [1]))
        return op.Mul(hidden_states, op.CastLike(op.Unsqueeze(valid, [1, 2]), hidden_states))

    def forward(
        self, op: OpBuilder, input_features: ir.Value, feature_lengths: ir.Value
    ) -> ir.Value:
        hidden_states = self._mask_padding(
            op, op.Unsqueeze(input_features, [1]), feature_lengths
        )  # (B, 1, mel_bins, T)
        hidden_states = self.act(op, self.conv_in(op, hidden_states))
        lengths = op.Div(op.Add(feature_lengths, 1), 2)
        hidden_states = self._mask_padding(op, hidden_states, lengths)
        hidden_states = self.act(op, self.lin_mid(op, self.conv_mid(op, hidden_states)))
        lengths = op.Div(op.Add(lengths, 1), 2)
        hidden_states = self._mask_padding(op, hidden_states, lengths)
        hidden_states = self.act(op, self.lin_end(op, self.conv_end(op, hidden_states)))
        # Merge channel and mel axes without collapsing batch or time.
        shape = op.Concat(
            op.Shape(hidden_states, start=0, end=1),
            [-1],
            op.Shape(hidden_states, start=3, end=4),
            axis=0,
        )
        return self.lin_out(op, op.Reshape(hidden_states, shape))


class _Lfm3AudioEncoderBackbone(nn.Module):
    """LFM2 over downsampled frames with causal sliding-window attention."""

    def __init__(self, config: Lfm2Config, sliding_window: int):
        super().__init__()
        self.config = apply_lfm2_config_defaults(config)
        if (
            self.config.layer_types is None
            or len(self.config.layer_types) != self.config.num_hidden_layers
            or set(self.config.layer_types) - {"conv", "full_attention"}
        ):
            raise ValueError(
                "LFM3 Audio encoder requires one 'conv' or 'full_attention' layer type "
                "per layer; use the checkpoint's audio encoder LFM2 layer_types."
            )
        self.sliding_window = sliding_window
        self.layers = nn.ModuleList(
            [
                Lfm2DecoderLayer(self.config, index)
                for index in range(self.config.num_hidden_layers)
            ]
        )
        self.embedding_norm = Lfm2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(self.config)
        if self.rotary_emb is None:
            raise ValueError(
                "LFM3 Audio encoder requires rotary position embeddings; "
                "use the checkpoint's LFM2 rope_parameters."
            )

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        num_frames = op.Squeeze(op.Shape(hidden_states, start=1, end=2), [0])
        positions = op.Range(0, num_frames, 1)
        position_ids = op.Expand(
            op.Unsqueeze(positions, [0]), op.Shape(hidden_states, start=0, end=2)
        )
        assert self.rotary_emb is not None
        position_embeddings = self.rotary_emb(op, position_ids)
        distance = op.Sub(op.Unsqueeze(positions, [1]), op.Unsqueeze(positions, [0]))
        allowed = op.And(
            op.GreaterOrEqual(distance, 0), op.Less(distance, self.sliding_window)
        )
        attention_bias = op.Unsqueeze(allowed, [0, 1])  # (1, 1, T, T)
        attention_mask = op.Expand(
            op.Constant(value_int=1), op.Shape(hidden_states, start=0, end=2)
        )
        layer_types = self.config.layer_types
        assert layer_types is not None
        for layer, kind in zip(self.layers, layer_types, strict=True):
            if kind == "conv":
                state_shape = op.Concat(
                    op.Shape(hidden_states, start=0, end=1),
                    [self.config.hidden_size, self.config.short_conv_kernel - 1],
                    axis=0,
                )
                state = op.Expand(op.CastLike(0.0, hidden_states), state_shape)
                past = (state,)
            else:
                # No cache is reused in the offline encoder.
                past = None
            hidden_states, _ = layer(
                op,
                hidden_states,
                attention_mask,
                attention_bias,
                position_embeddings,
                past,
            )
        return self.embedding_norm(op, hidden_states)


class Lfm3AudioEncoder(nn.Module):
    """Offline HuggingFace Lfm3AudioEncoder: log-mel frames to causal features.

    Inputs are (B, mel_bins, T) float32 log-mel features and (B,) int64
    valid lengths. Outputs are (B, ceil(T / 8), hidden_size) features and
    (B,) downsampled lengths; callers trim padding before placeholder fusion.
    """

    def __init__(
        self,
        config: Lfm2Config,
        *,
        num_mel_bins: int = 128,
        conv_channels: int = 256,
        sliding_window: int = 32,
    ):
        super().__init__()
        if sliding_window <= 0:
            raise ValueError(
                f"LFM3 Audio encoder sliding_window must be positive, got {sliding_window}; "
                "use the checkpoint's audio encoder configuration."
            )
        if config.dtype != ir.DataType.FLOAT:
            raise ValueError(
                "LFM3 Audio encoder currently requires float32 compute; "
                "construct its Lfm2Config with dtype=ir.DataType.FLOAT."
            )
        self.downsample = Lfm3AudioDownsampler(num_mel_bins, conv_channels, config.hidden_size)
        self.lfm = _Lfm3AudioEncoderBackbone(config, sliding_window)

    def forward(
        self, op: OpBuilder, input_features: ir.Value, feature_lengths: ir.Value
    ) -> tuple[ir.Value, ir.Value]:
        if input_features.dtype != ir.DataType.FLOAT:
            raise ValueError(
                "LFM3 Audio encoder input_features must be float32 log-mel features; "
                "retain the processor's float32 output at the graph boundary."
            )
        hidden_states = self.downsample(op, input_features, feature_lengths)
        hidden_states = self.lfm(op, op.Transpose(hidden_states, perm=[0, 2, 1]))
        return hidden_states, op.Div(op.Add(feature_lengths, 7), 8)


class Lfm3AudioProjector(nn.Module):
    """LayerNorm and exact-GELU MLP mapping encoder frames to language width."""

    def __init__(self, encoder_size: int, hidden_size: int):
        super().__init__()
        self.layer_norm = LayerNorm(encoder_size, eps=1e-5)
        self.linear_1 = Linear(encoder_size, hidden_size)
        self.linear_2 = Linear(hidden_size, hidden_size)

    def forward(self, op: OpBuilder, features: ir.Value) -> ir.Value:
        features = self.linear_1(op, self.layer_norm(op, features))
        return self.linear_2(op, op.Gelu(features, approximate="none"))


class Lfm3AudioProjectedEncoder(nn.Module):
    """Encode padded spectrograms and project valid-frame features for fusion."""

    def __init__(self, config: Lfm3AudioConfig):
        super().__init__()
        encoder = config.encoder_config
        if encoder is None:
            raise ValueError("LFM3 Audio needs encoder_config; load the composite config.")
        self.encoder = Lfm3AudioEncoder(
            dataclasses.replace(encoder, dtype=config.dtype),
            num_mel_bins=config.num_mel_bins,
            conv_channels=config.conv_channels,
            sliding_window=config.encoder_sliding_window,
        )
        self.projector = Lfm3AudioProjector(encoder.hidden_size, config.hidden_size)

    def forward(
        self, op: OpBuilder, input_features: ir.Value, feature_lengths: ir.Value
    ) -> tuple[ir.Value, ir.Value]:
        features, lengths = self.encoder(op, input_features, feature_lengths)
        return self.projector(op, features), lengths


class Lfm3AudioEmbedding(nn.Module):
    """Fuse packed input frames and output codebooks at global placeholder slots."""

    def __init__(self, config: Lfm3AudioConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.codebook_embedding = Embedding(
            config.num_codebooks * config.codebook_size, config.hidden_size
        )

    @staticmethod
    def _replace(
        op: OpBuilder,
        embeddings: ir.Value,
        input_ids: ir.Value,
        features: ir.Value,
        token_id: int,
    ) -> ir.Value:
        # Flatten across rows so each packed feature is consumed exactly once.
        mask = op.Equal(op.Reshape(input_ids, [-1]), token_id)
        offsets = op.Sub(op.CumSum(op.Cast(mask, to=ir.DataType.INT64), 0), 1)
        offsets = op.Max(offsets, 0)
        width = op.Shape(embeddings, start=2, end=3)
        padded = op.Concat(
            features,
            op.Expand(op.CastLike(0.0, features), op.Concat([1], width, axis=0)),
            axis=0,
        )
        selected = op.Gather(padded, offsets, axis=0)
        flat = op.Reshape(embeddings, op.Concat([-1], width, axis=0))
        return op.Reshape(
            op.Where(op.Unsqueeze(mask, [-1]), selected, flat), op.Shape(embeddings)
        )

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        audio_features: ir.Value,
        audio_out_codes: ir.Value,
    ) -> ir.Value:
        embeddings = self.embed_tokens(op, input_ids)
        embeddings = self._replace(
            op, embeddings, input_ids, audio_features, self.config.audio_token_id
        )
        offsets = [
            index * self.config.codebook_size for index in range(self.config.num_codebooks)
        ]
        code_embeddings = self.codebook_embedding(op, op.Add(audio_out_codes, offsets))
        output_features = op.ReduceSum(code_embeddings, [1], keepdims=False)
        return self._replace(
            op, embeddings, input_ids, output_features, self.config.audio_out_token_id
        )


class Lfm3AudioDecoder(nn.Module):
    """LFM3 decoder accepting fused embeddings and exposing speech-conditioning states."""

    def __init__(self, config: Lfm3AudioConfig):
        super().__init__()
        self.language_model = Lfm3TextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list,
    ):
        hidden, present = self.language_model(
            op,
            None,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds=inputs_embeds,
        )
        return self.lm_head(op, hidden), hidden, present


class _BidirectionalDepthAttention(Lfm3Attention):
    """LFM3 head-gated attention over all codebook slots, without a causal mask."""

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple,
    ):
        del attention_bias, past_key_value
        query, key, value = self._project_qkv(op, hidden_states)
        query = self.q_norm(op, op.Reshape(query, [0, 0, -1, self.head_dim]))
        key = self.k_norm(op, op.Reshape(key, [0, 0, -1, self.head_dim]))
        query = self._apply_rotary(
            op, op.Reshape(query, [0, 0, -1]), position_embeddings, self.num_attention_heads
        )
        key = self._apply_rotary(
            op, op.Reshape(key, [0, 0, -1]), position_embeddings, self.num_key_value_heads
        )
        attended = op.Attention(
            query,
            key,
            value,
            q_num_heads=self.num_attention_heads,
            kv_num_heads=self.num_key_value_heads,
            scale=self.scaling,
            is_causal=0,
        )
        gate = op.Unsqueeze(op.Sigmoid(self.gate_proj(op, hidden_states)), [-1])
        attended = op.Reshape(attended, [0, 0, self.num_attention_heads, self.head_dim])
        attended = op.Reshape(op.Mul(attended, gate), [0, 0, -1])
        return self.o_proj(op, attended), ()


class _DepthCodebook(nn.Module):
    def __init__(self, hidden_size: int, codebook_size: int, eps: float):
        super().__init__()
        self.embedding = Embedding(codebook_size, hidden_size)
        self.embedding_norm = Lfm3RMSNorm(hidden_size, eps=eps)

    def forward(
        self,
        op: OpBuilder,
        codes: ir.Value | None,
        hidden_states: ir.Value | None = None,
    ) -> ir.Value:
        if hidden_states is None:
            assert codes is not None
            return self.embedding(op, codes)
        normalized = self.embedding_norm(op, hidden_states)
        return op.MatMul(normalized, op.Transpose(self.embedding.weight, perm=[1, 0]))


class Lfm3AudioDepthDecoder(nn.Module):
    """Bidirectional LFM3 MaskGIT pass: frame state and masked codes to full logits."""

    def __init__(self, config: Lfm3AudioConfig):
        super().__init__()
        depth = config.depth_config
        if depth is None:
            raise ValueError("LFM3 Audio needs depth_config; load the composite config.")
        if set(depth.layer_types or []) != {"full_attention"}:
            raise ValueError(
                "LFM3 Audio depth decoder requires full_attention layers only; "
                "load the checkpoint's noncausal depth_config."
            )
        self.config = config
        self.depth_config = dataclasses.replace(depth, dtype=config.dtype)
        self.depth_linear = Linear(
            config.hidden_size, config.num_codebooks * depth.hidden_size
        )
        self.mask_embedding = nn.Parameter([depth.hidden_size])
        self.model = Lfm3TextModel(self.depth_config)
        for layer in self.model.layers:
            layer.self_attn = _BidirectionalDepthAttention(self.depth_config)
        self.depth_embeddings = nn.ModuleList(
            [
                _DepthCodebook(depth.hidden_size, config.codebook_size, depth.rms_norm_eps)
                for _ in range(config.num_codebooks)
            ]
        )

    def forward(
        self, op: OpBuilder, hidden_states: ir.Value, codes: ir.Value, masked: ir.Value
    ) -> ir.Value:
        slots = self.config.num_codebooks
        embeds = [
            op.Unsqueeze(embedding(op, op.Gather(codes, index, axis=1)), [1])
            for index, embedding in enumerate(self.depth_embeddings)
        ]
        code_embeddings = op.Concat(*embeds, axis=1)  # (frames, codebooks, depth_width)
        hidden = op.Reshape(
            self.depth_linear(op, hidden_states), [-1, slots, self.depth_config.hidden_size]
        )
        hidden = op.Add(
            hidden,
            op.Where(op.Unsqueeze(masked, [-1]), self.mask_embedding, code_embeddings),
        )
        # MatMulNBits inference does not propagate the projection's compute type.
        hidden.type = ir.TensorType(self.depth_config.dtype)
        shape = op.Shape(codes)
        attention_mask = op.Expand(op.Constant(value_int=1), shape)
        positions = op.Expand(op.Unsqueeze(op.Range(0, slots, 1), [0]), shape)
        hidden, _ = self.model(
            op,
            None,
            attention_mask,
            positions,
            [None] * self.depth_config.num_hidden_layers,
            inputs_embeds=hidden,
        )
        logits = []
        for index, embedding in enumerate(self.depth_embeddings):
            logits.append(
                op.Unsqueeze(
                    embedding(op, None, hidden_states=op.Gather(hidden, index, axis=1)),
                    [1],
                )
            )
        return op.Concat(*logits, axis=1)


class Lfm3AudioForConditionalGeneration(nn.Module):
    """LiquidAI LFM3 Audio's offline ASR and speech-code generation architecture.

    Replicates HuggingFace Lfm3AudioForConditionalGeneration as encoder,
    embedding, hybrid decoder, and bidirectional speech depth-decoder graphs.
    LiquidCodec converts the generated 12-codebook frames to 24-kHz audio.

    Float32 offline export only. ``build_lfm3_audio_package`` in
    ``mobius.integrations.lfm3_audio`` includes the separately stored codec;
    ``examples/lfm3_audio_generation.py`` demonstrates ASR and synthesis.
    """

    default_task: str = "lfm3-audio"
    category: str = "Audio"
    config_class: type = Lfm3AudioConfig
    preserve_standard_attention: bool = True

    def __init__(self, config: Lfm3AudioConfig):
        super().__init__()
        if config.dtype != ir.DataType.FLOAT:
            raise ValueError(
                "LFM3 Audio currently supports float32 compute only; export with dtype='f32'."
            )
        self.config = config
        self.audio_encoder = Lfm3AudioProjectedEncoder(config)
        self.embedding = Lfm3AudioEmbedding(config)
        self.decoder = Lfm3AudioDecoder(config)
        self.depth_decoder = Lfm3AudioDepthDecoder(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        prefixes = (
            ("lfm.model.", "model.language_model."),
            ("lfm.lm_head.", "lm_head."),
            ("encoder.downsample.", "model.audio_encoder.downsample."),
            ("encoder.lfm.lfm.", "model.audio_encoder.lfm."),
            ("mlp.model.0.", "model.audio_projector.layer_norm."),
            ("mlp.model.1.", "model.audio_projector.linear_1."),
            ("mlp.model.3.", "model.audio_projector.linear_2."),
            ("codebook_embedding.", "model.codebook_embedding."),
            ("depth_linear.", "model.depth_decoder.depth_linear."),
            ("depthformer.model.", "model.depth_decoder.model."),
            ("depth_embeddings.", "model.depth_decoder.depth_embeddings."),
        )
        weights = {}
        for name, value in state_dict.items():
            if name == "mask_embedding":
                name = "model.depth_decoder.mask_embedding"
            else:
                for source, target in prefixes:
                    if name.startswith(source):
                        name = target + name[len(source) :]
                        break
            if name.startswith("model.audio_encoder."):
                name = "audio_encoder.encoder." + name[len("model.audio_encoder.") :]
                name = rename_lfm2_weight_key(name)
            elif name.startswith("model.audio_projector."):
                name = "audio_encoder.projector." + name[len("model.audio_projector.") :]
            elif name == "model.language_model.embed_tokens.weight":
                name = "embedding.embed_tokens.weight"
            elif name.startswith("model.language_model."):
                name = "decoder.language_model." + name[len("model.language_model.") :]
            elif name.startswith("model.codebook_embedding."):
                name = (
                    "embedding.codebook_embedding." + name[len("model.codebook_embedding.") :]
                )
            elif name.startswith("model.depth_decoder."):
                name = "depth_decoder." + name[len("model.depth_decoder.") :]
            elif name == "lm_head.weight":
                name = "decoder.lm_head.weight"
            weights[name] = value
        if self.config.tie_word_embeddings and "decoder.lm_head.weight" not in weights:
            weights["decoder.lm_head.weight"] = weights["embedding.embed_tokens.weight"]
        return weights
