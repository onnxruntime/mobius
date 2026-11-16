# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Continuous diffusion drafting and causal verification for Nemotron-Labs-Sigma.

Replicates ``NemotronLabsSigmaModel`` with a shared decoder, two vocabulary
heads, and explicit clean-prefix caches. Draft calls return tentative caches;
the host must discard them. Verification caches may be trimmed to the accepted
prefix before encoding the bonus token.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import Attention, DecoderLayer, Embedding, Linear, RMSNorm

MODEL_ID = "nvidia/Nemotron-Labs-Sigma-distill-spec-3B"
REVISION = "9769b2d388000e0ff61736d26359074efbf0dc9d"


@dataclasses.dataclass
class NemotronSigmaConfig(ArchitectureConfig):
    """Sigma's native dual-head, projected-Bernstein checkpoint contract."""

    embed_dim: int = 16
    block_size: int = 32
    schedule_degree: int = 32
    original_context_length: int = 16384
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_attention_scale: float = 1.0
    long_context_beta: float = 0.1

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> NemotronSigmaConfig:
        for field, expected in {
            "include_ar_head": True,
            "native_family": "phased_dmd",
            "native_nfe": 2,
            "schedule_parameterization": "projected_bernstein",
            "replaid_clean_ar_embedding": True,
            "replaid_selfcond": True,
            "replaid_disable_time_conditioning": False,
            "replaid_disable_output_prior": False,
        }.items():
            actual = getattr(config, field, None)
            if actual != expected:
                raise ValueError(
                    f"Sigma export requires {field}={expected!r}, got {actual!r}. "
                    f"Use the supported dual-head checkpoint {MODEL_ID}@{REVISION}."
                )
        base = ArchitectureConfig.from_transformers(config, parent_config)
        fields = {field.name: getattr(base, field.name) for field in dataclasses.fields(base)}
        # Checkpoint storage is BF16; this route deliberately computes in FP32.
        fields["dtype"] = ir.DataType.FLOAT
        rope = config.rope_parameters
        factor = float(rope["factor"])

        def mscale(value: float) -> float:
            return 1.0 + 0.1 * value * math.log(factor) if factor > 1.0 else 1.0

        return cls(
            **fields,
            embed_dim=config.embed_dim,
            block_size=config.block_size,
            schedule_degree=config.schedule_degree,
            original_context_length=int(rope["original_max_position_embeddings"]),
            yarn_beta_fast=float(rope["beta_fast"]),
            yarn_beta_slow=float(rope["beta_slow"]),
            yarn_attention_scale=mscale(rope["mscale"]) / mscale(rope["mscale_all_dim"]),
            long_context_beta=float(rope["llama_4_scaling_beta"]),
        )

    def validate_sigma(self) -> None:
        if self.dtype != ir.DataType.FLOAT:
            raise ValueError(
                f"Sigma currently requires FLOAT compute, got {self.dtype}. "
                "Export with --dtype f32; reduced precision is not qualified."
            )
        if self.embed_dim <= 0 or self.block_size <= 0 or self.schedule_degree <= 0:
            raise ValueError(
                "Sigma embed_dim, block_size and schedule_degree must be positive; "
                "reload the pinned checkpoint config."
            )
        if self.head_dim % 2:
            raise ValueError(
                "Sigma RoPE needs an even head_dim; reload the checkpoint config."
            )


class _SigmaRotary(nn.Module):
    def __init__(self, config: NemotronSigmaConfig):
        super().__init__()
        dim = config.head_dim
        base = config.rope_theta
        original = config.original_context_length
        # Sigma uses max_context / original_context (0.25), NOT nominal factor 16.
        factor = config.max_position_embeddings / original

        def correction(rotations: float) -> float:
            return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(correction(config.yarn_beta_fast)), 0)
        high = min(math.ceil(correction(config.yarn_beta_slow)), dim - 1)
        ramp = np.clip(
            (np.arange(dim // 2, dtype=np.float32) - low) / max(high - low, 0.001), 0, 1
        )
        freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        self.frequency = (freq * (1 - ramp) + (freq / factor) * ramp).astype(np.float32)
        self.scale = config.yarn_attention_scale
        self.beta = config.long_context_beta
        self.original = original

    def forward(self, op: OpBuilder, position_ids, past_key):
        angles = op.Mul(
            op.Unsqueeze(op.Cast(position_ids, to=ir.DataType.FLOAT), [-1]),
            op.Constant(value=ir.tensor(self.frequency)),
        )
        cos = op.Mul(op.Cos(angles), self.scale)
        sin = op.Mul(op.Sin(angles), self.scale)
        start = op.Squeeze(op.Shape(past_key, start=2, end=3), [0])
        length = op.Squeeze(op.Shape(position_ids, start=1, end=2), [0])
        storage = op.Cast(op.Add(op.Range(0, length, 1), start), to=ir.DataType.FLOAT)
        scale = op.Add(
            1.0,
            op.Mul(
                self.beta,
                op.Log(op.Add(1.0, op.Floor(op.Div(storage, float(self.original))))),
            ),
        )
        return cos, sin, op.Unsqueeze(scale, [0, 2])


class _SigmaAttention(Attention):
    def forward(
        self,
        op,
        hidden_states,
        attention_bias,
        position_embeddings,
        past_key_value=None,
        static_cache=None,
    ):
        query = self.q_proj(op, hidden_states)
        key = self.k_proj(op, hidden_states)
        value = self.v_proj(op, hidden_states)
        cos, sin, scale = position_embeddings
        query = op.RotaryEmbedding(
            query, cos, sin, num_heads=self.num_attention_heads, interleaved=0
        )
        key = op.RotaryEmbedding(
            key, cos, sin, num_heads=self.num_key_value_heads, interleaved=0
        )
        query = op.Mul(query, scale)
        # The supplied mask carries causal verification OR bidirectional drafts.
        output, present_key, present_value = op.Attention(
            query,
            key,
            value,
            attention_bias,
            past_key_value[0],
            past_key_value[1],
            q_num_heads=self.num_attention_heads,
            kv_num_heads=self.num_key_value_heads,
            scale=self.scaling,
            is_causal=0,
            _outputs=3,
        )
        return self.o_proj(op, output), (present_key, present_value)


class _SigmaEncoder(nn.Module):
    def __init__(self, config: NemotronSigmaConfig):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        layers = []
        for _ in range(config.num_hidden_layers):
            layer = DecoderLayer(config)
            layer.self_attn = _SigmaAttention(config)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = _SigmaRotary(config)

    def forward(
        self, op, input_ids, noisy, denoise, position_ids, attention_mask, past_key_values
    ):
        clean = self.embed_tokens(op, input_ids)
        hidden = op.Where(denoise, noisy, clean)  # (B, L, hidden)
        positions = self.rotary_emb(op, position_ids, past_key_values[0][0])
        present = []
        for layer, past in zip(self.layers, past_key_values):
            hidden, kv = layer(op, hidden, attention_mask, positions, past)
            present.append(kv)
        return self.norm(op, hidden), present


class _SigmaEmbeddingMatrix(nn.Module):
    def __init__(self, config: NemotronSigmaConfig):
        super().__init__()
        self.matrix = nn.Parameter([config.vocab_size, config.embed_dim])

    def forward(self, op):
        norm = op.Sqrt(op.ReduceSum(op.Mul(self.matrix, self.matrix), [1], keepdims=1))
        return op.Div(self.matrix, op.Add(norm, 1e-8))


class _SigmaGammaBounds(nn.Module):
    def __init__(self):
        super().__init__()
        self.endpoint_values = nn.Parameter([2])

    def forward(self, op):
        return op.Identity(self.endpoint_values)


class NemotronSigmaModel(nn.Module):
    """Sigma continuous diffusion drafts and autoregressive verification.

    A shared ONNX graph accepts tokens or 16-D noisy embeddings, a scalar noise
    level, an explicit attention mask, and clean-prefix KV caches. It emits both
    vocabulary heads, normalized denoising means and tentative present caches.
    Host speculative decoding commits only a verified prefix and its bonus token.
    Only the pinned Phased-DMD dual-head contract and FP32 export are supported.
    """

    default_task = "nemotron-sigma"
    category = "Text Generation"
    config_class = NemotronSigmaConfig

    def __init__(self, config: NemotronSigmaConfig):
        super().__init__()
        config.validate_sigma()
        self.config = config
        self.encoder = _SigmaEncoder(config)
        self.diffusion_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.ar_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.replaid_embedding_matrix = _SigmaEmbeddingMatrix(config)
        self.replaid_input_linear = Linear(config.embed_dim, config.hidden_size, bias=False)
        self.replaid_selfcond_linear = Linear(config.embed_dim, config.hidden_size, bias=False)
        self.replaid_gamma_linear = Linear(64, config.hidden_size, bias=False)
        self.gamma_bounds = _SigmaGammaBounds()

    def forward(
        self,
        op,
        input_ids,
        z,
        gamma,
        self_condition,
        denoise,
        position_ids,
        attention_mask,
        past_key_values,
    ):
        self.config.validate_sigma()
        embedding = self.replaid_embedding_matrix(op)
        alpha_sq = op.Sigmoid(op.Neg(gamma))
        sigma_sq = op.Sigmoid(gamma)
        variance = op.Add(op.Div(alpha_sq, float(self.config.embed_dim)), sigma_sq)
        x_in = op.Cast(op.Div(z, op.Sqrt(variance)), to=ir.DataType.FLOAT)
        noisy = self.replaid_input_linear(op, x_in)
        condition = self.replaid_selfcond_linear(
            op, op.Mul(self_condition, math.sqrt(self.config.embed_dim))
        )
        frequencies = np.exp(np.linspace(-5, 5, 32, dtype=np.float32)).astype(np.float32)
        angles = op.Mul(
            op.Cast(gamma, to=ir.DataType.FLOAT), op.Constant(value=ir.tensor(frequencies))
        )
        time = self.replaid_gamma_linear(op, op.Concat(op.Sin(angles), op.Cos(angles), axis=0))
        noisy = op.Add(op.Add(noisy, condition), time)
        hidden, present = self.encoder(
            op, input_ids, noisy, denoise, position_ids, attention_mask, past_key_values
        )
        logits = self.diffusion_head(op, hidden)
        scaled_z = op.Mul(
            op.Cast(op.Div(op.Sqrt(alpha_sq), sigma_sq), to=ir.DataType.FLOAT),
            op.Cast(z, to=ir.DataType.FLOAT),
        )
        prior = op.MatMul(scaled_z, op.Transpose(embedding, perm=[1, 0]))
        logits = op.Add(logits, prior)
        mean = op.MatMul(op.Softmax(logits, axis=-1), embedding)
        return logits, mean, self.ar_head(op, hidden), self.gamma_bounds(op), present
