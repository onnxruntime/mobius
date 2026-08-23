# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Typed ``package`` facts for onnx-genai inference metadata.

A published package carries neural graphs and the workflow that drives them, but
a request arrives as text and media.  Turning that request into the token stream
the graphs consume requires the vocabulary contract the package was built
against: which algorithm produced the ids, how many ids exist, and which id
plays each semantic role.  onnx-genai calls that section ``package.tokenizer``
(``PackageFacts``/``TokenizerFacts`` in ``onnx-genai-metadata``), and it is the
only place a front end looks for those facts.

Two of those roles are load-bearing rather than informational:

``eos``
    The workflow's termination policy already compares generated ids against a
    stop id.  Publishing the same id under a role means a caller can render and
    trim a transcript without re-deriving it from a side file.

``image_placeholder``
    The prompt token whose position an image's features replace.  A multimodal
    package that omits it declares no place in the token stream for its own
    image features, so an attached image is preprocessed and then dropped.

Every value here is read from the package's own artifacts.  The algorithm,
vocabulary size and byte-level flag come from the packaged tokenizer definition;
each special token pairs a config-declared id with the surface form that id has
in that same vocabulary.  A role the package does not declare is simply absent,
because a guessed special token is worse than a missing one.

Nothing in this module knows a model name or a literal token id: architecture
defaults belong to the config adapters under :mod:`mobius._configs`, and reach
the emitters as ordinary config attributes.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

from mobius.integrations.onnx_genai.inference_metadata import _source_asset_path

_LOGGER = logging.getLogger(__name__)

#: Semantic role naming the prompt token that stands for one whole image.
#: Fixed by the onnx-genai runtime, which looks this role up by name.
IMAGE_PLACEHOLDER_ROLE: Final = "image_placeholder"


@dataclasses.dataclass(frozen=True)
class SpecialTokenRole:
    """One semantic role and the config fields that may carry its id.

    ``fields`` are HuggingFace-standard config attribute names.  Architectures
    that name a role differently normalize it in their config adapter, so this
    table never grows a model-specific entry.
    """

    name: str
    fields: tuple[str, ...]


#: Roles every text-producing package can state about its own vocabulary.
TEXT_TOKEN_ROLES: Final[tuple[SpecialTokenRole, ...]] = (
    SpecialTokenRole("bos", ("bos_token_id",)),
    SpecialTokenRole("eos", ("eos_token_id",)),
    SpecialTokenRole("pad", ("pad_token_id",)),
    SpecialTokenRole("unk", ("unk_token_id",)),
)

#: Roles a package adds when a modality's features replace a prompt token.
MEDIA_TOKEN_ROLES: Final[tuple[SpecialTokenRole, ...]] = (
    SpecialTokenRole(IMAGE_PLACEHOLDER_ROLE, ("image_token_id",)),
    SpecialTokenRole("audio_placeholder", ("audio_token_id",)),
    SpecialTokenRole("video_placeholder", ("video_token_id",)),
)

#: Nested config objects searched after the root for a role's id.  A composite
#: architecture keeps its text and vision ids in sub-configs under these names.
_CONFIG_SCOPES: Final[tuple[str, ...]] = (
    "text",
    "text_config",
    "vision",
    "vision_config",
    "audio",
    "audio_config",
)

#: ``tokenizer.json`` model types mapped to the schema's algorithm vocabulary.
_ALGORITHMS: Final[Mapping[str, str]] = {
    "BPE": "bpe",
    "Unigram": "unigram",
    "WordPiece": "wordpiece",
    "WordLevel": "word_level",
}

#: Package-relative tokenizer assets, in the order a package lists them.
#: Only the ones the source actually provides are declared.
TOKENIZER_ARTIFACT_NAMES: Final[tuple[str, ...]] = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)


@dataclasses.dataclass(frozen=True)
class SpecialTokenFact:
    """One special token, pinned by id and exact surface bytes."""

    id: int
    content: str

    def to_metadata(self) -> dict[str, Any]:
        return {"id": self.id, "content": self.content}


@dataclasses.dataclass(frozen=True)
class TokenizerArtifact:
    """One package-relative tokenizer artifact."""

    location: str

    def to_metadata(self) -> dict[str, Any]:
        return {"location": self.location}


@dataclasses.dataclass(frozen=True)
class TokenizerFacts:
    """Tokenizer facts and package-relative artifacts.

    Mirrors onnx-genai's ``TokenizerFacts``.  ``algorithm`` and ``vocab_size``
    are required by that contract; the rest are omitted when empty rather than
    published as defaults.
    """

    algorithm: str
    vocab_size: int
    byte_level: bool = False
    artifacts: tuple[TokenizerArtifact, ...] = ()
    special_tokens: Mapping[str, SpecialTokenFact] = dataclasses.field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        facts: dict[str, Any] = {
            "algorithm": self.algorithm,
            "vocab_size": self.vocab_size,
            "byte_level": self.byte_level,
        }
        if self.artifacts:
            facts["artifacts"] = [artifact.to_metadata() for artifact in self.artifacts]
        if self.special_tokens:
            facts["special_tokens"] = {
                role: token.to_metadata()
                for role, token in sorted(self.special_tokens.items())
            }
        return facts


@dataclasses.dataclass(frozen=True)
class PackageFacts:
    """Exact package facts required to interpret request data correctly."""

    tokenizer: TokenizerFacts | None = None

    def to_metadata(self) -> dict[str, Any]:
        facts: dict[str, Any] = {}
        if self.tokenizer is not None:
            facts["tokenizer"] = self.tokenizer.to_metadata()
        return facts


@dataclasses.dataclass(frozen=True)
class TokenizerDefinition:
    """The packaged tokenizer's own description of its vocabulary.

    ``surface_forms`` maps a vocabulary id to the exact bytes that id renders
    as, which is what pins a special token to more than a bare number.
    """

    algorithm: str
    byte_level: bool
    surface_forms: Mapping[int, str]

    @property
    def vocab_size(self) -> int:
        """Number of vocabulary entries, including added tokens.

        Added tokens are appended above the base vocabulary and may leave gaps,
        so the highest occupied id plus one is the width a caller must assume,
        not the number of entries present.
        """
        return max(self.surface_forms) + 1 if self.surface_forms else 0


def _load_json(path: str) -> Any:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        _LOGGER.warning("Could not read package asset %s", path)
        return None


def _uses_byte_level(node: Any) -> bool:
    """Whether any component round-trips raw bytes rather than Unicode scalars."""
    if isinstance(node, dict):
        if node.get("type") in {"ByteLevel", "ByteFallback"}:
            return True
        return any(_uses_byte_level(value) for value in node.values())
    if isinstance(node, list):
        return any(_uses_byte_level(value) for value in node)
    return False


def _surface_forms(vocab: Any, added_tokens: Any) -> dict[int, str]:
    """Map vocabulary id to surface form; added tokens take precedence."""
    surfaces: dict[int, str] = {}
    if isinstance(vocab, dict):
        for token, index in vocab.items():
            if isinstance(index, int) and not isinstance(index, bool):
                surfaces[index] = str(token)
    elif isinstance(vocab, list):
        # A Unigram vocabulary is ``[[piece, score], ...]`` with position as id.
        for index, entry in enumerate(vocab):
            if isinstance(entry, (list, tuple)) and entry:
                surfaces[index] = str(entry[0])
    if isinstance(added_tokens, list):
        for added in added_tokens:
            if isinstance(added, dict) and isinstance(added.get("id"), int):
                content = added.get("content")
                if content is not None:
                    surfaces[int(added["id"])] = str(content)
    return surfaces


def read_tokenizer_definition(source: str | None) -> TokenizerDefinition | None:
    """Read the packaged tokenizer's own vocabulary description.

    ``tokenizer.json`` is a complete tokenizer definition and is preferred: it
    states the algorithm, the whole vocabulary and the normalizer/pre-tokenizer
    chain that decides whether ids address bytes or characters.

    A package whose tokenizer predates that format ships the artifacts instead:
    a flat ``vocab.json``, plus ``merges.txt`` when and only when the vocabulary
    is a merge table.  That pair is itself the evidence for the algorithm — a
    vocabulary with merges is BPE, one without is a flat word-level table where
    each entry is one unit — so it is read rather than guessed at.

    Returns ``None`` when no tokenizer definition is reachable.  A package that
    cannot show its vocabulary states no tokenizer facts at all, because a
    plausible-looking guess would be indistinguishable from a measured value.
    """
    if not source:
        return None

    path = _source_asset_path(source, "tokenizer.json")
    if path is not None:
        definition = _load_json(path)
        if isinstance(definition, dict):
            model = definition.get("model")
            model = model if isinstance(model, dict) else {}
            algorithm = _ALGORITHMS.get(str(model.get("type", "")))
            surfaces = _surface_forms(model.get("vocab"), definition.get("added_tokens"))
            if algorithm is not None and surfaces:
                return TokenizerDefinition(
                    algorithm=algorithm,
                    # A byte-level tokenizer can declare that in any stage of
                    # its chain, so every stage is inspected.
                    byte_level=any(
                        _uses_byte_level(definition.get(section))
                        for section in ("pre_tokenizer", "decoder", "model", "normalizer")
                    ),
                    surface_forms=surfaces,
                )

    vocab_path = _source_asset_path(source, "vocab.json")
    if vocab_path is None:
        return None
    vocab = _load_json(vocab_path)
    surfaces = _surface_forms(vocab, None)
    if not surfaces:
        return None
    has_merges = _source_asset_path(source, "merges.txt") is not None
    return TokenizerDefinition(
        algorithm="bpe" if has_merges else "word_level",
        # Legacy artifacts carry no normalizer chain, so there is nothing that
        # states byte addressing; the schema default of ``false`` is the only
        # honest answer.
        byte_level=False,
        surface_forms=surfaces,
    )


def _config_value(config: Any, field: str) -> Any:
    """Read ``field`` from the config, then from its nested sub-configs."""
    value = getattr(config, field, None)
    if value is not None:
        return value
    for scope in _CONFIG_SCOPES:
        nested = getattr(config, scope, None)
        if nested is None:
            continue
        value = getattr(nested, field, None)
        if value is not None:
            return value
    return None


def _token_id(value: Any) -> int | None:
    """Coerce a declared token id, taking the first of a stop-id list."""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


_MISSING: Final = object()


def source_declared_value(source: str | None, name: str, default: Any = _MISSING) -> Any:
    """Read a value the package's own runtime documents declare.

    ``genai_config.json`` is the package's record of how it is served and wins
    over ``tokenizer_config.json``; both outrank a value derived from the
    architecture config, because a repackaged checkpoint may have been retuned
    without its config being rewritten.
    """
    if not source or not os.path.isdir(source):
        return default
    candidates: tuple[tuple[str, tuple[str, ...]], ...] = (
        (os.path.join(source, "genai_config.json"), ("model", name)),
        (os.path.join(source, "tokenizer_config.json"), (name,)),
    )
    for path, keys in candidates:
        if not os.path.isfile(path):
            continue
        value = _load_json(path)
        try:
            for key in keys:
                value = value[key]
        except (TypeError, KeyError, IndexError):
            continue
        return value
    return default


def source_declared_roles(
    source: str | None,
    roles: Iterable[SpecialTokenRole],
) -> dict[str, Any]:
    """Role ids the package's own runtime documents declare.

    The emitted workflow resolves its stop id the same way, so reading the same
    documents here keeps one document from stating one id for its termination
    policy and a different id for the same role.
    """
    declared: dict[str, Any] = {}
    for role in roles:
        for field in role.fields:
            value = source_declared_value(source, field, None)
            if value is not None:
                declared[role.name] = value
                break
    return declared


def build_special_tokens(
    definition: TokenizerDefinition,
    config: Any,
    roles: Iterable[SpecialTokenRole],
    *,
    declared: Mapping[str, Any] | None = None,
) -> dict[str, SpecialTokenFact]:
    """Resolve declared roles against the packaged vocabulary.

    ``declared`` overrides the config for roles the emitting workflow has
    already resolved for itself, so a document never states one id for its
    termination policy and a different one for the same role here.

    A role survives only when both halves of the fact are known: an id the
    package declares *and* the exact surface bytes that id has in this
    vocabulary.  An id with no surface form is a stale config field pointing
    outside the shipped vocabulary, and publishing it would let a front end
    splice a token the tokenizer cannot render.
    """
    declared = declared or {}
    special_tokens: dict[str, SpecialTokenFact] = {}
    for role in roles:
        token_id = _token_id(declared.get(role.name))
        if token_id is None:
            for field in role.fields:
                token_id = _token_id(_config_value(config, field))
                if token_id is not None:
                    break
        if token_id is None:
            continue
        content = definition.surface_forms.get(token_id)
        if not content:
            continue
        special_tokens[role.name] = SpecialTokenFact(id=token_id, content=content)
    return special_tokens


def build_tokenizer_facts(
    source: str | None,
    config: Any,
    *,
    roles: Sequence[SpecialTokenRole] = TEXT_TOKEN_ROLES,
) -> TokenizerFacts | None:
    """Describe the packaged tokenizer, or ``None`` when it cannot be read.

    Args:
        source: HuggingFace model id or local directory carrying the package's
            tokenizer artifacts.
        config: Resolved architecture config; supplies each role's id.
        roles: Semantic roles this package can state.  A text package states
            text roles; a multimodal one adds the media placeholders whose
            positions its encoder features replace.

    Returns:
        The facts, or ``None`` when no tokenizer definition is reachable.

    ``vocab_size`` is the tokenizer's own vocabulary width rather than the
    decoder's logits width.  The two usually agree; where they differ the logits
    tensor is padded to a hardware-friendly multiple and the extra columns
    address nothing, so the vocabulary is the width a caller can actually render.

    Artifact locations are deliberately not derived from ``source``: they are
    package-relative paths, and only the package directory says which files it
    ships.  :func:`declare_tokenizer_artifacts` fills them in once it exists.
    """
    definition = read_tokenizer_definition(source)
    if definition is None or definition.vocab_size <= 0:
        return None
    return TokenizerFacts(
        algorithm=definition.algorithm,
        vocab_size=definition.vocab_size,
        byte_level=definition.byte_level,
        special_tokens=build_special_tokens(
            definition,
            config,
            roles,
            declared=source_declared_roles(source, roles),
        ),
    )


def attach_package_facts(
    metadata: dict[str, Any],
    source: str | None,
    config: Any,
    *,
    roles: Sequence[SpecialTokenRole] = TEXT_TOKEN_ROLES,
) -> None:
    """Merge tokenizer facts into ``metadata['package']`` when they are known.

    The section is left absent when nothing can be read, so a reader can tell
    "this package does not state its tokenizer" apart from "this package states
    an empty tokenizer".
    """
    tokenizer = build_tokenizer_facts(source, config, roles=roles)
    if tokenizer is None:
        return
    package = metadata.setdefault("package", {})
    package.setdefault("tokenizer", tokenizer.to_metadata())


def declare_tokenizer_artifacts(metadata: dict[str, Any], package_dir: str) -> None:
    """Declare the tokenizer files the package directory actually contains.

    ``TokenizerArtifact.location`` is a package-relative path a reader may open,
    so the only thing that can answer it is the package itself.  Deriving it
    from the source checkpoint instead would name files that the writer never
    copied — a CTC package ships ``vocab.json`` where a BPE package ships
    ``tokenizer.json``, and neither ships the other.

    Does nothing when the document states no tokenizer facts to attach them to.
    """
    tokenizer = (metadata.get("package") or {}).get("tokenizer")
    if not isinstance(tokenizer, dict):
        return
    locations = [
        {"location": name}
        for name in TOKENIZER_ARTIFACT_NAMES
        if os.path.isfile(os.path.join(package_dir, name))
    ]
    if locations:
        tokenizer["artifacts"] = locations
    else:
        tokenizer.pop("artifacts", None)
