# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Non-generative CLM and Kev model components and deterministic adapters."""

from __future__ import annotations

import dataclasses
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components._common import LayerNorm, Linear
from mobius.models.qwen import Qwen3CausalLMModel
from mobius.models.qwen35 import Qwen35CausalLMModel

JSONContent = str | int | float | bool | None | list["JSONContent"] | dict[str, "JSONContent"]

CLM_MODEL_ID = "Contrastive-LM/CLM-v0.1-8B"
CLM_REVISION = "e939398d4556fcd9400c76fa8c5a513202f42b0a"
CLM_BASE_MODEL_ID = "Qwen/Qwen3-8B"
KEV_MODEL_ID = "jaredpalmer/kev-4b"
KEV_REVISION = "139fdd94f1b6a6ad80cc15e08fcb99cac885a101"
KEV_BASE_MODEL_ID = "Qwen/Qwen3.5-4B-Base"
KEV_BASE_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
KEV_HIDDEN_SIZE = 2560
KEV_08_MODEL_ID = "jaredpalmer/kev-0.8b"
KEV_08_REVISION = "bf75a6a8848ea6960ff2ed108d9ed44c2941174f"
KEV_08_BASE_MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
KEV_08_BASE_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
KEV_08_HIDDEN_SIZE = 1024
KEV_POINTER_SIZE = 256
KEV_TEMPERATURE = 2.406050072164233
KEV_08_TEMPERATURE = 2.3510958125672174
KEV_MAX_OPTIONS = 255
KEV_MAX_STATE_TOKENS = 8192
KEV_MAX_ROW_TOKENS = 8192
CLM_HIDDEN_SIZE = 4096
CLM_PROJECTION_DIM = 512
CLM_WIDTH = 1536


@dataclasses.dataclass(frozen=True)
class ModelProvenance:
    """Immutable identities needed to reproduce an exported decision model."""

    model_id: str
    revision: str
    base_model_id: str
    base_revision: str | None

    def as_metadata(self) -> dict[str, str]:
        """Return string metadata, marking an absent base revision unpinned."""
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "base_model_id": self.base_model_id,
            "base_revision": self.base_revision or "unpinned",
            "reproducible": str(self.base_revision is not None).lower(),
        }


CLM_PROVENANCE = ModelProvenance(CLM_MODEL_ID, CLM_REVISION, CLM_BASE_MODEL_ID, None)
KEV_PROVENANCE = ModelProvenance(
    KEV_MODEL_ID, KEV_REVISION, KEV_BASE_MODEL_ID, KEV_BASE_REVISION
)
KEV_08_PROVENANCE = ModelProvenance(
    KEV_08_MODEL_ID,
    KEV_08_REVISION,
    KEV_08_BASE_MODEL_ID,
    KEV_08_BASE_REVISION,
)


@dataclasses.dataclass(frozen=True)
class KevVariant:
    """Immutable model-specific KEV export contract."""

    name: str
    provenance: ModelProvenance
    hidden_size: int
    temperature: float


KEV_VARIANTS = {
    KEV_BASE_MODEL_ID: KevVariant(
        "kev-4b",
        KEV_PROVENANCE,
        KEV_HIDDEN_SIZE,
        KEV_TEMPERATURE,
    ),
    KEV_08_BASE_MODEL_ID: KevVariant(
        "kev-0.8b",
        KEV_08_PROVENANCE,
        KEV_08_HIDDEN_SIZE,
        KEV_08_TEMPERATURE,
    ),
}


class CheckpointContractError(ValueError):
    """A checkpoint does not match the pinned model-specific export contract."""


def _metadata_matches(actual: Any, expected: Any) -> bool:
    """Compare checkpoint metadata without accepting bool/int coercions."""
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, int):
        return isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
    if isinstance(expected, float):
        return isinstance(actual, float) and actual == expected
    return isinstance(actual, type(expected)) and actual == expected


def _validate_weight_mapping(weights: Mapping[str, Any], *, owner: str) -> None:
    """Require a non-empty string-to-tensor weight mapping."""
    if not isinstance(weights, Mapping) or not weights:
        raise CheckpointContractError(f"{owner} weights must be a non-empty mapping")
    invalid = [
        key
        for key, value in weights.items()
        if not isinstance(key, str) or not isinstance(value, torch.Tensor)
    ]
    if invalid:
        raise CheckpointContractError(
            f"{owner} weights must contain only string-to-tensor entries"
        )


def _apply_package_weights_strict(
    package, weights: dict[str, torch.Tensor], *, owner: str
) -> None:
    """Apply every supplied weight and require every initializer to be bound."""
    applied = package.apply_weights(weights)
    unapplied = sorted(set(weights) - applied)
    if unapplied:
        examples = ", ".join(repr(name) for name in unapplied[:5])
        suffix = f" and {len(unapplied) - 5} more" if len(unapplied) > 5 else ""
        raise CheckpointContractError(
            f"{owner} contains unapplied weights: {examples}{suffix}"
        )
    missing = [
        f"{component}.{name}"
        for component, model in package.items()
        for name, initializer in model.graph.initializers.items()
        if initializer.const_value is None
    ]
    if missing:
        examples = ", ".join(repr(name) for name in missing[:5])
        suffix = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
        raise CheckpointContractError(
            f"{owner} leaves initializers without weights: {examples}{suffix}"
        )


def _require_tensor_shape(
    state_dict: Mapping[str, Any], key: str, shape: tuple[int, ...], *, owner: str
) -> None:
    """Require one checkpoint tensor with an exact name, type, and shape."""
    value = state_dict.get(key)
    if not isinstance(value, torch.Tensor):
        raise CheckpointContractError(f"{owner}.{key} must be a torch.Tensor")
    if tuple(value.shape) != shape:
        raise CheckpointContractError(
            f"{owner}.{key} shape must be {shape}, got {tuple(value.shape)}"
        )


def validate_clm_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    """Fail closed unless *checkpoint* is the published CLM-v0.1 head layout."""
    if not isinstance(checkpoint, Mapping):
        raise CheckpointContractError("CLM checkpoint must be a mapping")
    cfg = checkpoint.get("cfg")
    if not isinstance(cfg, Mapping):
        raise CheckpointContractError("CLM checkpoint must contain a cfg mapping")
    expected = {
        "model": CLM_BASE_MODEL_ID,
        "hidden_size": CLM_HIDDEN_SIZE,
        "width": CLM_WIDTH,
        "depth": 3,
        "activation": "gelu",
        "layernorm": True,
        "residual": False,
    }
    for key, value in expected.items():
        if key not in cfg or not _metadata_matches(cfg[key], value):
            raise CheckpointContractError(
                f"CLM checkpoint cfg.{key} must be {value!r}, got {cfg.get(key)!r}"
            )
    projection_dim = checkpoint.get("projection_dim", cfg.get("projection_dim"))
    if not _metadata_matches(projection_dim, CLM_PROJECTION_DIM):
        raise CheckpointContractError("CLM checkpoint projection_dim must be 512")
    shapes = {
        "inp.weight": (CLM_WIDTH, CLM_HIDDEN_SIZE),
        "inp.bias": (CLM_WIDTH,),
        "hidden.0.weight": (CLM_WIDTH, CLM_WIDTH),
        "hidden.0.bias": (CLM_WIDTH,),
        "norms.0.weight": (CLM_WIDTH,),
        "norms.0.bias": (CLM_WIDTH,),
        "out.weight": (CLM_PROJECTION_DIM, CLM_WIDTH),
        "out.bias": (CLM_PROJECTION_DIM,),
    }
    for name in ("state_head", "action_head"):
        head = checkpoint.get(name)
        if not isinstance(head, Mapping):
            raise CheckpointContractError(f"CLM checkpoint must contain a {name} state dict")
        if set(head) != set(shapes):
            missing = sorted(map(str, set(shapes) - set(head)))
            extra = sorted(map(str, set(head) - set(shapes)))
            raise CheckpointContractError(
                f"CLM {name} keys mismatch; missing={missing}, extra={extra}"
            )
        for key, shape in shapes.items():
            _require_tensor_shape(head, key, shape, owner=name)
    logit_scale = checkpoint.get("logit_scale")
    if isinstance(logit_scale, bool) or not isinstance(
        logit_scale, (int, float, torch.Tensor)
    ):
        raise CheckpointContractError("CLM logit_scale must be a scalar")
    if isinstance(logit_scale, torch.Tensor) and logit_scale.numel() != 1:
        raise CheckpointContractError("CLM logit_scale must be a scalar")


def validate_kev_checkpoint(checkpoint: Mapping[str, Any]) -> KevVariant:
    """Validate and return the exact published KEV variant contract."""
    if not isinstance(checkpoint, Mapping):
        raise CheckpointContractError("Kev checkpoint must be a mapping")
    base = checkpoint.get("base")
    variant = KEV_VARIANTS.get(base)
    if variant is None:
        raise CheckpointContractError(
            f"Kev checkpoint base must be one of {sorted(KEV_VARIANTS)}, got {base!r}"
        )
    expected = {
        "base": variant.provenance.base_model_id,
        "base_revision": variant.provenance.base_revision,
        "head_dim": KEV_POINTER_SIZE,
        "option_isolation": False,
        "temperature": variant.temperature,
    }
    for key, value in expected.items():
        actual = checkpoint.get(key)
        if not _metadata_matches(actual, value):
            raise CheckpointContractError(
                f"Kev checkpoint {key} must be {value!r}, got {actual!r}"
            )
    head = checkpoint.get("head")
    if not isinstance(head, Mapping):
        raise CheckpointContractError("Kev checkpoint must contain a head state dict")
    shapes = {
        "q.weight": (KEV_POINTER_SIZE, variant.hidden_size),
        "q.bias": (KEV_POINTER_SIZE,),
        "k.weight": (KEV_POINTER_SIZE, variant.hidden_size),
        "k.bias": (KEV_POINTER_SIZE,),
    }
    if set(head) != set(shapes):
        missing = sorted(map(str, set(shapes) - set(head)))
        extra = sorted(map(str, set(head) - set(shapes)))
        raise CheckpointContractError(
            f"Kev head keys mismatch; missing={missing}, extra={extra}"
        )
    for key, shape in shapes.items():
        _require_tensor_shape(head, key, shape, owner="head")
    return variant


def render_clm(value: JSONContent, indent: int = 0) -> str:
    """Render structured CLM input exactly as the reference preprocessing does."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    pad = " " * indent
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            if isinstance(item, (dict, list)) and item:
                parts.append(f"{pad}{key}:\n{render_clm(item, indent + 2)}")
            else:
                parts.append(f"{pad}{key}: {render_clm(item)}")
        return ("\n\n" if indent == 0 else "\n").join(parts)
    parts = []
    for item in value:
        if isinstance(item, (dict, list)) and item:
            parts.append(f"{pad}-\n{render_clm(item, indent + 2)}")
        else:
            parts.append(f"{pad}- {render_clm(item)}")
    return "\n".join(parts)


def clm_state_text(state: JSONContent, instructions: JSONContent) -> str:
    """Render state and instructions with the reference blank-line separator."""
    state_value = render_clm(state).strip()
    instruction_value = render_clm(instructions).strip()
    return (
        f"{state_value}\n\n{instruction_value}"
        if state_value and instruction_value
        else state_value or instruction_value
    )


def clm_candidates(question: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Return answer keys and exact action-head candidate strings."""
    question_type = question.get("type")
    criteria = question.get("criteria")
    instructions = render_clm(question.get("instructions")).strip()
    if question_type == "choice":
        if not isinstance(criteria, Mapping) or not criteria:
            raise ValueError("choice question needs a non-empty 'criteria' object")
        keys = list(criteria)
        return keys, [
            render_clm(criteria[key]) if criteria[key] not in (None, "") else key
            for key in keys
        ]
    if question_type == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("score question needs an ordered list of at least two levels")
        return [str(i) for i in range(len(criteria))], [render_clm(item) for item in criteria]
    if question_type != "noul":
        raise ValueError(f"unknown question type {question_type!r}")
    criteria = criteria or {}
    values = []
    for key in ("false", "true"):
        description = criteria.get(key) if isinstance(criteria, Mapping) else None
        if description in (None, ""):
            if instructions:
                description = (
                    f"Yes. This is true: {instructions}"
                    if key == "true"
                    else f"No. This is false: {instructions}"
                )
            else:
                description = key
        values.append(f"{key}: {render_clm(description)}")
    return ["false", "true"], values


def clm_pairs(
    state: JSONContent, questions: Mapping[str, Mapping[str, Any]]
) -> dict[str, tuple[str, list[str], list[str]]]:
    """Build each question's state text, answer keys, and action texts."""
    return {
        key: (
            clm_state_text(state, question.get("instructions")),
            *clm_candidates(question),
        )
        for key, question in questions.items()
    }


def clm_last_token_indices(attention_mask: Sequence[Sequence[int]]) -> list[int]:
    """Return each row's last attended token for padding-aware CLM pooling."""
    indices = []
    for row_index, row in enumerate(attention_mask):
        attended = [index for index, value in enumerate(row) if value]
        if not attended:
            raise ValueError(f"attention_mask row {row_index} has no attended token")
        indices.append(attended[-1])
    return indices


def validate_temperature(temperature: float) -> float:
    """Return a valid request temperature or reject values outside ``(0, 100]``."""
    value = float(temperature)
    if not 0.0 < value <= 100.0:
        raise ValueError("temperature must be in (0, 100]")
    return value


def stable_softmax(values: Sequence[float]) -> list[float]:
    """Compute overflow-safe softmax for one non-empty logit group."""
    if not values:
        raise ValueError("softmax group must not be empty")
    maximum = max(values)
    exponentials = [math.exp(float(value) - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def grouped_softmax(
    logits: Sequence[float], group_sizes: Sequence[int], *, temperature: float = 1.0
) -> list[list[float]]:
    """Split flat logits into validated groups and softmax each independently."""
    temperature = validate_temperature(temperature)
    if any(size <= 0 for size in group_sizes) or sum(group_sizes) != len(logits):
        raise ValueError("group_sizes must be positive and consume every logit")
    result, offset = [], 0
    for size in group_sizes:
        result.append(
            stable_softmax([value / temperature for value in logits[offset : offset + size]])
        )
        offset += size
    return result


def rank_probabilities(
    candidates: Sequence[str], probabilities: Sequence[float]
) -> list[dict]:
    """Rank candidates by descending probability with stable input-order ties."""
    if len(candidates) != len(probabilities):
        raise ValueError("candidates and probabilities must have equal lengths")
    order = sorted(range(len(candidates)), key=lambda index: (-probabilities[index], index))
    return [
        {
            "rank": rank + 1,
            "candidate": candidates[index],
            "prob": float(probabilities[index]),
        }
        for rank, index in enumerate(order)
    ]


def clm_answer(
    question: Mapping[str, Any], keys: Sequence[str], probabilities: Sequence[float]
) -> dict[str, Any]:
    """Map one CLM distribution to its exact noul/choice/score answer."""
    probs = [float(value) for value in probabilities]
    if len(keys) != len(probs) or not probs:
        raise ValueError("keys and probabilities must have the same non-zero length")
    question_type = question["type"]
    distribution = dict(zip(keys, probs))
    if question_type == "noul":
        return {"type": "noul", "noul": distribution["true"]}
    confidence = _clm_confidence(probs)
    if question_type == "choice":
        selected = max(range(len(probs)), key=probs.__getitem__)
        return {
            "type": "choice",
            "choice": keys[selected],
            "confidence": confidence,
            "probabilities": distribution,
        }
    levels = question["criteria"]
    return {
        "type": "score",
        "score": sum(index * value for index, value in enumerate(probs)),
        "confidence": confidence,
        "legend": {str(i): render_clm(level) for i, level in enumerate(levels)},
        "probabilities": distribution,
    }


def _clm_confidence(probabilities: Sequence[float]) -> float:
    """Compute CLM's top probability minus the mean competing probability."""
    if len(probabilities) < 2:
        return 1.0
    selected = max(range(len(probabilities)), key=probabilities.__getitem__)
    rest = [value for index, value in enumerate(probabilities) if index != selected]
    return max(0.0, min(1.0, probabilities[selected] - sum(rest) / len(rest)))


class CLMProjectionHead(nn.Module):
    """The checkpoint-compatible CLM ``hidden -> projection`` MLP."""

    def __init__(
        self,
        *,
        hidden_size: int = 4096,
        width: int,
        depth: int = 3,
        projection_dim: int = 512,
        activation: Literal["gelu", "relu", "silu"] = "gelu",
        layernorm: bool = True,
        residual: bool = False,
    ):
        """Construct the checkpoint-shaped projection MLP.

        ``depth`` includes input and output projections, so depth three creates
        exactly one hidden block. Unsupported depths and activations fail
        before or during graph construction.
        """
        super().__init__()
        if depth < 2:
            raise ValueError("CLM projection depth must be at least 2")
        self.inp = Linear(hidden_size, width)
        self.hidden = nn.ModuleList([Linear(width, width) for _ in range(depth - 2)])
        self.norms = nn.ModuleList(
            [
                LayerNorm(width, eps=1e-5) if layernorm else _Identity()
                for _ in range(depth - 2)
            ]
        )
        self.out = Linear(width, projection_dim)
        self.activation = activation
        self.residual = residual

    def _activate(self, op: OpBuilder, value: ir.Value) -> ir.Value:
        """Apply the configured reference activation to one graph value."""
        if self.activation == "gelu":
            return op.Gelu(value)
        if self.activation == "relu":
            return op.Relu(value)
        if self.activation == "silu":
            return op.Mul(value, op.Sigmoid(value))
        raise ValueError(f"unsupported CLM activation {self.activation!r}")

    def forward(self, op: OpBuilder, embeddings: ir.Value) -> ir.Value:
        """Project ``[items, hidden]`` embeddings and L2-normalize each row."""
        embeddings_float = op.Cast(embeddings, to=ir.DataType.FLOAT)
        input_norm = op.ReduceL2(embeddings_float, [-1], keepdims=1)
        normalized = op.CastLike(
            op.Div(embeddings_float, op.Max(input_norm, op.CastLike(1e-12, input_norm))),
            embeddings,
        )
        value = self._activate(op, self.inp(op, normalized))
        for linear, norm in zip(self.hidden, self.norms):
            projected = self._activate(op, norm(op, linear(op, value)))
            value = op.Add(value, projected) if self.residual else projected
        value = self.out(op, value)
        value_float = op.Cast(value, to=ir.DataType.FLOAT)
        norm = op.ReduceL2(value_float, [-1], keepdims=1)
        return op.CastLike(
            op.Div(value_float, op.Max(norm, op.CastLike(1e-12, norm))),
            value,
        )


class _Identity(nn.Module):
    """Graph identity used where a checkpoint omits hidden LayerNorm."""

    def forward(self, op: OpBuilder, value: ir.Value) -> ir.Value:
        """Return an ONNX identity of *value*."""
        return op.Identity(value)


class CLMScorer(nn.Module):
    """Scaled-cosine scorer; inputs are already L2-normalized projections."""

    def __init__(self):
        """Create the learned scalar log-temperature parameter."""
        super().__init__()
        self.logit_scale = nn.Parameter([1])

    def forward(
        self,
        op: OpBuilder,
        states: ir.Value,
        actions: ir.Value,
        candidate_owners: ir.Value,
        temperature: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        """Score owned state/action rows and return grouped logits/probabilities."""
        scale = op.Min(op.Exp(self.logit_scale), op.CastLike(100.0, self.logit_scale))
        state_rows = op.Gather(states, candidate_owners, axis=0)
        scores = op.ReduceSum(op.Mul(state_rows, actions), [-1], keepdims=0)
        logits = op.Div(op.Mul(scores, scale), temperature)
        return logits, _grouped_softmax_graph(op, logits, candidate_owners, states)


class CLMModel(nn.Module):
    """Qwen3-8B encoder and both CLM-v0.1 projection heads."""

    default_task = "clm-scoring"
    category = "Text Ranking"
    provenance = CLM_PROVENANCE

    def __init__(
        self,
        config: ArchitectureConfig,
        *,
        width: int,
        depth: int = 3,
        projection_dim: int = 512,
        activation: Literal["gelu", "relu", "silu"] = "gelu",
        layernorm: bool = True,
        residual: bool = False,
        base_revision: str | None = None,
    ):
        """Construct a headless Qwen3 encoder and exact CLM-v0.1 heads.

        ``base_revision`` is optional for manual graph construction; package
        provenance explicitly reports such a graph as unpinned. The public
        production builder requires a concrete revision.
        """
        super().__init__()
        if config.hidden_size != 4096:
            raise ValueError("CLM-v0.1-8B requires Qwen3-8B hidden_size=4096")
        self.config = config
        self.provenance = dataclasses.replace(CLM_PROVENANCE, base_revision=base_revision)
        self.encoder = Qwen3EncoderModel(config)
        options = {
            "hidden_size": config.hidden_size,
            "width": width,
            "depth": depth,
            "projection_dim": projection_dim,
            "activation": activation,
            "layernorm": layernorm,
            "residual": residual,
        }
        self.state_head = CLMProjectionHead(**options)
        self.action_head = CLMProjectionHead(**options)
        self.scorer = CLMScorer()

    def preprocess_weights(self, state_dict: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Map combined base/head tensors into this module's component paths."""
        mapped = map_clm_checkpoint(state_dict)
        base = {
            key: value
            for key, value in state_dict.items()
            if isinstance(key, str)
            and isinstance(value, torch.Tensor)
            and not key.startswith(("state_head.", "action_head.", "scorer."))
            and key != "logit_scale"
        }
        for key, value in self.encoder.preprocess_weights(base).items():
            mapped.setdefault(f"encoder.{key}", value)
        return mapped


def map_clm_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Flatten a validated in-memory CLM checkpoint without deserializing code."""
    mapped: dict[str, torch.Tensor] = {}
    for source, target in (("state_head", "state_head"), ("action_head", "action_head")):
        values = checkpoint.get(source)
        if values is None:
            continue
        if not isinstance(values, Mapping):
            raise TypeError(f"{source} must be a state-dict mapping")
        for key, value in values.items():
            if not isinstance(key, str) or not isinstance(value, torch.Tensor):
                raise TypeError(f"{source} must contain string-to-tensor entries")
            mapped[f"{target}.{key}"] = value
    scale = checkpoint.get("logit_scale")
    if scale is not None:
        mapped["scorer.logit_scale"] = torch.as_tensor(scale).reshape(1)
    for key, value in checkpoint.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            continue
        if key.startswith(("state_head.", "action_head.", "scorer.", "encoder.")):
            mapped[key] = value
    return mapped


KEV_CONTROL_TOKEN_IDS = {
    "state": 248060,
    "question": 248061,
    "option_start": 248049,
    "option_end": 248050,
    "decide": 248062,
}
_KEV_DELIMITER_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def escape_kev_delimiters(text: str) -> str:
    """Rewrite Qwen control-token spellings so user text cannot forge boundaries."""
    return _KEV_DELIMITER_RE.sub(r"<¦\1¦>", text)


def render_kev(value: JSONContent, indent: int = 0) -> str:
    """Kev rendering, including Python's capitalized bool spelling."""
    pad = "  " * indent
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(f"{pad}- {render_kev(item, indent + 1).lstrip()}" for item in value)
    return "\n".join(
        (
            f"{pad}{key}:\n{render_kev(item, indent + 1)}"
            if isinstance(item, (dict, list))
            else f"{pad}{key}: {render_kev(item)}"
        )
        for key, item in value.items()
    )


def kev_question_options(question: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Return exact Kev answer keys/texts, enforcing type and 1-255 bounds."""
    question_type = question.get("type")
    criteria = question.get("criteria")
    if question_type == "noul":
        criteria = criteria or {}
        return ["false", "true"], [
            _kev_option_text("no", criteria.get("false")),
            _kev_option_text("yes", criteria.get("true")),
        ]
    if question_type == "choice":
        if not isinstance(criteria, Mapping) or not 1 <= len(criteria) <= KEV_MAX_OPTIONS:
            raise ValueError("choice question needs 1..255 criteria")
        keys = list(criteria)
        return keys, [_kev_option_text(key, criteria[key]) for key in keys]
    if question_type == "score":
        if not isinstance(criteria, list) or not 1 <= len(criteria) <= KEV_MAX_OPTIONS:
            raise ValueError("score question needs 1..255 levels")
        return [str(i) for i in range(len(criteria))], [render_kev(item) for item in criteria]
    raise ValueError(f"unknown question type {question_type!r}")


def _kev_option_text(name: str, description: JSONContent) -> str:
    """Render an option key alone or followed by its structured description."""
    return name if description in (None, "") else f"{name}: {render_kev(description)}"


@dataclasses.dataclass(frozen=True)
class KevQuestionRow:
    """One independent causal state+question row and its readout indices."""

    question_id: str
    input_ids: tuple[int, ...]
    position_ids: tuple[int, ...]
    decide_index: int
    option_indices: tuple[int, ...]
    keys: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class KevBatch:
    """Right-padded backbone inputs and pointer-head indices."""

    input_ids: tuple[tuple[int, ...], ...]
    attention_mask: tuple[tuple[int, ...], ...]
    position_ids: tuple[tuple[int, ...], ...]
    decide_indices: tuple[int, ...]
    option_indices: tuple[int, ...]
    option_owners: tuple[int, ...]


def encode_kev_rows(
    tokenizer: Callable[..., Any],
    state: JSONContent,
    questions: Mapping[str, Mapping[str, Any]],
    *,
    strict: bool = False,
    max_state_tokens: int = KEV_MAX_STATE_TOKENS,
    max_row_tokens: int = KEV_MAX_ROW_TOKENS,
) -> tuple[KevQuestionRow, ...]:
    """Encode one causal row/question with the verified serving limits.

    The state control token counts toward ``max_state_tokens``. In normal
    serving mode user state tokens are truncated to ``max_state_tokens - 1``;
    strict mode rejects the same overflow instead.
    """
    if not questions:
        raise ValueError("questions must not be empty")
    if max_state_tokens < 1 or max_row_tokens < max_state_tokens:
        raise ValueError("token limits must be positive and row >= state")

    def tokens(text: str) -> list[int]:
        """Tokenize escaped caller text without tokenizer-added special tokens."""
        encoded = tokenizer(escape_kev_delimiters(text), add_special_tokens=False)
        values = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
        return list(values)

    user_state_ids = tokens(render_kev(state))
    if strict and len(user_state_ids) + 1 > max_state_tokens:
        raise ValueError(f"state exceeds {max_state_tokens} tokens")
    state_ids = [
        KEV_CONTROL_TOKEN_IDS["state"],
        *user_state_ids[: max_state_tokens - 1],
    ]
    rows = []
    for question_id, question in questions.items():
        keys, options = kev_question_options(question)
        # Each question starts a fresh causal row that repeats the bounded state.
        ids = [
            *state_ids,
            KEV_CONTROL_TOKEN_IDS["question"],
            *tokens(render_kev(question.get("instructions"))),
        ]
        option_indices = []
        for option in options:
            ids.extend([KEV_CONTROL_TOKEN_IDS["option_start"], *tokens(option)])
            ids.append(KEV_CONTROL_TOKEN_IDS["option_end"])
            # Pointer K reads the closing boundary, matching Kev training.
            option_indices.append(len(ids) - 1)
        ids.append(KEV_CONTROL_TOKEN_IDS["decide"])
        if len(ids) > max_row_tokens:
            raise ValueError(f"state+question row exceeds {max_row_tokens} tokens: {len(ids)}")
        # The final decide token is the pointer Q location for this row.
        rows.append(
            KevQuestionRow(
                question_id,
                tuple(ids),
                tuple(range(len(ids))),
                len(ids) - 1,
                tuple(option_indices),
                tuple(keys),
            )
        )
    return tuple(rows)


def batch_kev_rows(rows: Sequence[KevQuestionRow], *, pad_token_id: int) -> KevBatch:
    """Right-pad encoded rows and flatten their aligned pointer readouts."""
    if not rows:
        raise ValueError("rows must not be empty")
    width = max(len(row.input_ids) for row in rows)
    input_ids, attention_mask, position_ids = [], [], []
    option_indices, option_owners = [], []
    for owner, row in enumerate(rows):
        padding = width - len(row.input_ids)
        input_ids.append((*row.input_ids, *((pad_token_id,) * padding)))
        attention_mask.append((*((1,) * len(row.input_ids)), *((0,) * padding)))
        position_ids.append((*row.position_ids, *((0,) * padding)))
        option_indices.extend(row.option_indices)
        option_owners.extend([owner] * len(row.option_indices))
    return KevBatch(
        tuple(input_ids),
        tuple(attention_mask),
        tuple(position_ids),
        tuple(row.decide_index for row in rows),
        tuple(option_indices),
        tuple(option_owners),
    )


def _normalize_probabilities(values: Sequence[float]) -> list[float]:
    """Normalize values, using a uniform distribution when their sum is zero."""
    total = sum(values)
    return (
        [1.0 / len(values)] * len(values)
        if total == 0
        else [float(value) / total for value in values]
    )


def kev_choice_confidence(probabilities: Sequence[float]) -> float:
    """Compute Kev's uniform-adjusted maximum-probability confidence."""
    probs = _normalize_probabilities(probabilities)
    count = len(probs)
    return 1.0 if count == 1 else (max(probs) - 1 / count) / (1 - 1 / count)


def kev_score_confidence(probabilities: Sequence[float]) -> float:
    """Compute public score confidence from expected distance to the mode."""
    probs = _normalize_probabilities(probabilities)
    count = len(probs)
    if count == 1:
        return 1.0
    mode = max(range(count), key=probs.__getitem__)
    return max(
        0.0,
        1.0
        - sum(value * abs(index - mode) for index, value in enumerate(probs)) / (count - 1),
    )


def _round4(value: float) -> float:
    """Round a response scalar to Kev's four-decimal wire precision."""
    return round(float(value), 4)


def kev_answer(
    question: Mapping[str, Any], keys: Sequence[str], probabilities: Sequence[float]
) -> dict[str, Any]:
    """Convert a Kev option distribution into exact noul/choice/score output."""
    probs = [float(value) for value in probabilities]
    question_type = question["type"]
    if question_type == "noul":
        return {"type": "noul", "noul": _round4(probs[1])}
    distribution = {key: _round4(value) for key, value in zip(keys, probs)}
    if question_type == "choice":
        index = max(range(len(probs)), key=probs.__getitem__)
        return {
            "type": "choice",
            "choice": keys[index],
            "confidence": _round4(kev_choice_confidence(probs)),
            "probabilities": distribution,
        }
    return {
        "type": "score",
        "score": _round4(sum(index * value for index, value in enumerate(probs))),
        "legend": {str(i): render_kev(item) for i, item in enumerate(question["criteria"])},
        "probabilities": distribution,
        "confidence": _round4(kev_score_confidence(probs)),
    }


class KevPointerHead(nn.Module):
    """KEV checkpoint-compatible 256-dimensional pointer head."""

    def __init__(
        self,
        hidden_size: int = KEV_HIDDEN_SIZE,
        temperature: float = KEV_TEMPERATURE,
    ):
        """Create checkpoint-compatible Q/K projections and fixed calibration."""
        super().__init__()
        self.q = Linear(hidden_size, KEV_POINTER_SIZE)
        self.k = Linear(hidden_size, KEV_POINTER_SIZE)
        self.temperature = temperature

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        decide_indices: ir.Value,
        option_indices: ir.Value,
        option_owners: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        """Read indexed row tokens and return calibrated grouped scores.

        ``decide_indices`` and ``option_indices`` are positions local to each
        independently padded question row; ``option_owners`` maps each option
        to the corresponding row.
        """
        # Form [row, token] coordinates so GatherND cannot cross question rows.
        question_ids = op.Range(
            op.Constant(value_int=0),
            op.Gather(op.Shape(decide_indices), 0),
            op.Constant(value_int=1),
        )
        decide_coordinates = op.Concat(
            op.Unsqueeze(question_ids, [1]),
            op.Unsqueeze(decide_indices, [1]),
            axis=1,
        )
        option_coordinates = op.Concat(
            op.Unsqueeze(option_owners, [1]),
            op.Unsqueeze(option_indices, [1]),
            axis=1,
        )
        decide = op.GatherND(hidden_states, decide_coordinates)
        options = op.GatherND(hidden_states, option_coordinates)
        queries = op.Gather(self.q(op, decide), option_owners, axis=0)
        logits = op.ReduceSum(op.Mul(self.k(op, options), queries), [-1], keepdims=0)
        logits = op.Div(logits, op.CastLike(math.sqrt(KEV_POINTER_SIZE), logits))
        logits = op.Div(logits, op.CastLike(self.temperature, logits))
        probabilities = _grouped_softmax_graph(op, logits, option_owners, decide_indices)
        return logits, probabilities


def _grouped_softmax_graph(
    op: OpBuilder,
    logits: ir.Value,
    owners: ir.Value,
    group_rows: ir.Value,
) -> ir.Value:
    """Build stable softmax over flat logits independently for every owner.

    Owner ids must be contiguous zero-based group indices. A one-hot ownership
    matrix computes each group's maximum and denominator without mixing
    questions.
    """
    question_count = op.Gather(op.Shape(group_rows), 0)
    mask = op.OneHot(
        owners,
        question_count,
        [0.0, 1.0],
        axis=-1,
    )
    # Subtract each owner's maximum before Exp, then gather its own denominator.
    expanded = op.Unsqueeze(logits, [1])
    masked = op.Where(
        op.Cast(mask, to=ir.DataType.BOOL),
        expanded,
        op.CastLike(-3.4028234663852886e38, logits),
    )
    maxima = op.ReduceMax(masked, [0], keepdims=0)
    stable = op.Sub(logits, op.Gather(maxima, owners))
    exponentials = op.Exp(stable)
    weighted_mask = op.CastLike(mask, exponentials)
    totals = op.MatMul(
        op.Transpose(weighted_mask, perm=[1, 0]),
        op.Unsqueeze(exponentials, [1]),
    )
    totals = op.Squeeze(totals, [1])
    return op.Div(exponentials, op.Gather(totals, owners))


class KevModel(nn.Module):
    """Pinned Qwen3.5 backbone and KEV pointer readout."""

    default_task = "kev-scoring"
    category = "Text Classification"

    def __init__(
        self,
        config: ArchitectureConfig,
        variant: KevVariant | None = None,
    ):
        """Construct the verified-width headless Qwen3.5 and pointer head."""
        super().__init__()
        variant = variant or KEV_VARIANTS[KEV_BASE_MODEL_ID]
        if config.hidden_size != variant.hidden_size:
            raise ValueError(
                f"{variant.name} requires Qwen3.5 hidden_size={variant.hidden_size}"
            )
        self.provenance = variant.provenance
        self.config = config
        self.backbone = Qwen35EncoderModel(config)
        self.pointer_head = KevPointerHead(config.hidden_size, variant.temperature)

    def preprocess_weights(self, state_dict: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Map merged backbone and pointer tensors into component paths."""
        mapped = map_kev_checkpoint(state_dict)
        base = {
            key.removeprefix("backbone."): value
            for key, value in state_dict.items()
            if isinstance(key, str)
            and isinstance(value, torch.Tensor)
            and not key.startswith("pointer_head.")
        }
        for key, value in self.backbone.preprocess_weights(base).items():
            mapped.setdefault(f"backbone.{key}", value)
        return mapped


def map_kev_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Map controlled head/base tensors; callers deserialize ``head.pt`` themselves."""
    mapped: dict[str, torch.Tensor] = {}
    head = checkpoint.get("head")
    if head is not None:
        if not isinstance(head, Mapping):
            raise TypeError("Kev checkpoint head must be a state-dict mapping")
        for key, value in head.items():
            if key not in {"q.weight", "q.bias", "k.weight", "k.bias"}:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError("Kev head values must be tensors")
            mapped[f"pointer_head.{key}"] = value
    for key, value in checkpoint.items():
        if isinstance(key, str) and isinstance(value, torch.Tensor):
            if key.startswith("backbone."):
                mapped[key] = value
            elif key.startswith("model."):
                mapped[f"backbone.{key}"] = value
    return mapped


def synthetic_clm_checkpoint(
    *,
    hidden_size: int,
    width: int,
    depth: int = 3,
    projection_dim: int = 512,
    seed: int = 0,
) -> dict[str, Any]:
    """Create a flexible deterministic test fixture, not a production checkpoint."""
    generator = torch.Generator().manual_seed(seed)

    def tensor(*shape: int) -> torch.Tensor:
        """Draw one deterministic tensor from the fixture generator."""
        return torch.randn(shape, generator=generator)

    def head() -> dict[str, torch.Tensor]:
        """Create one fixture head using checkpoint-compatible key names."""
        values = {
            "inp.weight": tensor(width, hidden_size),
            "inp.bias": tensor(width),
            "out.weight": tensor(projection_dim, width),
            "out.bias": tensor(projection_dim),
        }
        for index in range(depth - 2):
            values[f"hidden.{index}.weight"] = tensor(width, width)
            values[f"hidden.{index}.bias"] = tensor(width)
            values[f"norms.{index}.weight"] = tensor(width)
            values[f"norms.{index}.bias"] = tensor(width)
        return values

    return {
        "state_head": head(),
        "action_head": head(),
        "logit_scale": torch.tensor(0.0),
        "cfg": {
            "hidden_size": hidden_size,
            "width": width,
            "depth": depth,
            "projection_dim": projection_dim,
        },
    }


def synthetic_kev_checkpoint(
    *, hidden_size: int = KEV_HIDDEN_SIZE, seed: int = 0
) -> dict[str, Any]:
    """Create flexible deterministic Kev test weights, not a production artifact."""
    generator = torch.Generator().manual_seed(seed)
    return {
        "head": {
            "q.weight": torch.randn(KEV_POINTER_SIZE, hidden_size, generator=generator),
            "q.bias": torch.randn(KEV_POINTER_SIZE, generator=generator),
            "k.weight": torch.randn(KEV_POINTER_SIZE, hidden_size, generator=generator),
            "k.bias": torch.randn(KEV_POINTER_SIZE, generator=generator),
        },
        "temperature": KEV_TEMPERATURE,
    }


def provenance_json(provenance: ModelProvenance) -> str:
    """Serialize deterministic compact provenance for ONNX metadata."""
    return json.dumps(provenance.as_metadata(), sort_keys=True, separators=(",", ":"))


def build_clm_package(
    config: ArchitectureConfig,
    *,
    base_weights: Mapping[str, torch.Tensor],
    head_checkpoint: Mapping[str, Any],
    base_revision: str,
    execution_provider: str = "default",
):
    """Build and populate CLM-v0.1 from separate base and head checkpoints.

    ``head_checkpoint`` must already be loaded by trusted caller code. The
    upstream CLM artifact does not pin Qwen3-8B, so a concrete base revision is
    mandatory here and is embedded into every component's provenance.

    Validation is fail-closed and occurs before graph construction: the exact
    published metadata, head key sets, tensor shapes, and scalar logit scale
    are required. Base weights must be a non-empty tensor mapping.
    """
    if not base_revision or not base_revision.strip():
        raise ValueError(
            "CLM export requires the caller to supply the exact Qwen3-8B base_revision"
        )
    _validate_weight_mapping(base_weights, owner="CLM base")
    # Validate the entire artifact contract before allocating a large graph.
    validate_clm_checkpoint(head_checkpoint)
    cfg = head_checkpoint.get("cfg")
    assert isinstance(cfg, Mapping)
    module = CLMModel(
        config,
        width=CLM_WIDTH,
        depth=3,
        projection_dim=CLM_PROJECTION_DIM,
        activation="gelu",
        layernorm=True,
        residual=False,
        base_revision=base_revision,
    )
    from mobius._builder import build_from_module
    from mobius.tasks._decision import CLMTask

    package = build_from_module(
        module,
        config,
        task=CLMTask(module.provenance),
        execution_provider=execution_provider,
    )
    weights = {
        f"encoder.{key}": value
        for key, value in module.encoder.preprocess_weights(dict(base_weights)).items()
    }
    weights.update(map_clm_checkpoint(head_checkpoint))
    # The component graphs retain their root module paths (for example,
    # ``encoder.model.layers...`` and ``state_head.inp...``). Trying to route by
    # those same prefixes would strip them and leave every initializer unbound.
    # Let ModelPackage match the fully qualified names across all components.
    _apply_package_weights_strict(package, weights, owner="CLM checkpoint")
    return package


def build_kev_package(
    config: ArchitectureConfig,
    *,
    head_checkpoint: Mapping[str, Any],
    merged_base_weights: Mapping[str, torch.Tensor] | None = None,
    base_weights: Mapping[str, torch.Tensor] | None = None,
    peft_adapter_weights: Mapping[str, torch.Tensor] | None = None,
    execution_provider: str = "default",
):
    """Build Kev from a correctly PEFT-merged Qwen3.5 state dict plus ``head.pt``.

    Mobius does not currently merge arbitrary PEFT LoRA artifacts into dense
    Qwen3.5 tensors. Supplying separate base/adapter mappings therefore fails
    explicitly; merge with PEFT at the pinned base revision first and pass the
    resulting state dict as ``merged_base_weights``.

    The published base identity, pointer metadata, calibration, exact Q/K key
    set, and tensor shapes are validated before graph construction. Separate
    PEFT/base inputs fail explicitly rather than producing an unadapted model.
    """
    # A valid head cannot make an unmerged PEFT base safe; enforce both gates.
    variant = validate_kev_checkpoint(head_checkpoint)
    if merged_base_weights is None:
        if base_weights is not None or peft_adapter_weights is not None:
            raise NotImplementedError(
                "Kev export cannot merge separate PEFT LoRA weights yet; load "
                f"{variant.provenance.base_model_id}@"
                f"{variant.provenance.base_revision}, merge the adapter with "
                "PEFT merge_and_unload(), and pass merged_base_weights"
            )
        raise ValueError("merged_base_weights is required for Kev export")
    if base_weights is not None or peft_adapter_weights is not None:
        raise ValueError(
            "pass either merged_base_weights or separate base/adapter inputs, not both"
        )
    _validate_weight_mapping(merged_base_weights, owner="Kev merged base")
    module = KevModel(config, variant)
    from mobius._builder import build_from_module
    from mobius.tasks._decision import KevTask

    package = build_from_module(
        module,
        config,
        task=KevTask(),
        execution_provider=execution_provider,
    )
    weights = {
        f"backbone.{key}": value
        for key, value in module.backbone.preprocess_weights(dict(merged_base_weights)).items()
    }
    weights.update(map_kev_checkpoint(head_checkpoint))
    # As with CLM, these graphs retain ``backbone.`` and ``pointer_head.`` in
    # their initializer names, so applying the fully qualified mapping directly
    # is required.
    _apply_package_weights_strict(package, weights, owner="Kev checkpoint")
    return package


class Qwen3EncoderModel(Qwen3CausalLMModel):
    """Headless Qwen3 backbone returning every token hidden state and KV state."""

    def __init__(self, config: ArchitectureConfig):
        """Initialize Qwen3 then remove the unused vocabulary projection."""
        super().__init__(config)
        del self.lm_head

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        """Return full ``[batch, sequence, hidden]`` states and KV cache.

        No pooling is performed: callers must select the last attended token,
        because right padding makes a fixed position ``-1`` incorrect.
        """
        # Keep every token state; CLM pooling depends on the caller's padding mask.
        return self.model(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Apply Qwen3 preprocessing while discarding any tied LM-head tensor."""
        mapped = super().preprocess_weights(state_dict)
        mapped.pop("lm_head.weight", None)
        return mapped


class Qwen35EncoderModel(Qwen35CausalLMModel):
    """Headless Qwen3.5 backbone returning token hidden states and hybrid state."""

    def __init__(self, config: ArchitectureConfig):
        """Initialize Qwen3.5 then remove the unused vocabulary projection."""
        super().__init__(config)
        del self.lm_head

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        """Return full token hidden states and Qwen3.5 hybrid cache state."""
        # Kev reads arbitrary decide/option positions, so token states stay unpooled.
        return self.model(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Apply Qwen3.5 preprocessing while discarding LM-head weights."""
        mapped = super().preprocess_weights(state_dict)
        mapped.pop("lm_head.weight", None)
        return mapped
