# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Typed ``package`` facts for onnx-genai inference metadata.

A published package carries neural graphs and the workflow that drives them, but
a request arrives as text and media.  Turning that request into the token stream
the graphs consume requires the vocabulary contract the package was built
against: which algorithm produced the ids, how many ids exist, and which numeric ids
play each execution-relevant role.  onnx-genai calls that section ``package.tokenizer``
(``PackageFacts``/``TokenizerFacts`` in ``onnx-genai-metadata``), and it is the
only place a front end looks for those facts.

Two of those facts are load-bearing rather than informational:

``eos_token_id``
    The workflow's termination policy already compares generated ids against a
    stop id.  Publishing the same id under a role means a caller can render and
    trim a transcript without re-deriving it from a side file.

``image_token_id``
    The prompt token whose position an image's features replace.  A multimodal
    package that omits it declares no place in the token stream for its own
    image features, so an attached image is preprocessed and then dropped.

Every value here is read from the package's own artifacts or resolved config.
The algorithm, vocabulary size and byte-level flag come from the packaged
tokenizer definition; numeric token IDs come from the package's runtime config.
Token spellings and chat templates remain authoritative in tokenizer assets,
so execution metadata never carries a second, potentially stale vocabulary.

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

#: Schema field naming the prompt token that stands for one whole image.
IMAGE_PLACEHOLDER_ROLE: Final = "image_token_id"


@dataclasses.dataclass(frozen=True)
class SpecialTokenRole:
    """One semantic role and the config fields that may carry its id.

    ``fields`` are HuggingFace-standard config attribute names.  Architectures
    that name a role differently normalize it in their config adapter, so this
    table never grows a model-specific entry.
    """

    name: str
    fields: tuple[str, ...]
    multiple: bool = False


#: Roles every text-producing package can state about its own vocabulary.
TEXT_TOKEN_ROLES: Final[tuple[SpecialTokenRole, ...]] = (
    SpecialTokenRole("pad_token_id", ("pad_token_id",)),
    SpecialTokenRole("bos_token_id", ("bos_token_id",)),
    SpecialTokenRole("eos_token_id", ("eos_token_id",), multiple=True),
    SpecialTokenRole("sep_token_id", ("sep_token_id",)),
    SpecialTokenRole("decoder_start_token_id", ("decoder_start_token_id",)),
)

#: Roles a package adds when a modality's features replace a prompt token.
MEDIA_TOKEN_ROLES: Final[tuple[SpecialTokenRole, ...]] = (
    SpecialTokenRole(IMAGE_PLACEHOLDER_ROLE, ("image_token_id",)),
    SpecialTokenRole("video_token_id", ("video_token_id",)),
    SpecialTokenRole("audio_token_id", ("audio_token_id",)),
    SpecialTokenRole("vision_start_token_id", ("vision_start_token_id",)),
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
#: The tokenizer subset of ``_RUNTIME_ASSET_NAMES``: every file the writers can
#: copy into a package, so a declared artifact set is never missing one they did.
TOKENIZER_ARTIFACT_NAMES: Final[tuple[str, ...]] = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "chat_template.jinja",
)


@dataclasses.dataclass(frozen=True)
class SpecialTokenFact:
    """Compatibility value for callers of the former text-bearing API.

    New metadata uses :class:`TokenFacts`; token content remains in tokenizer
    assets. Keeping this value exported avoids breaking existing imports.
    """

    id: int
    content: str

    def to_metadata(self) -> dict[str, Any]:
        return {"id": self.id, "content": self.content}


@dataclasses.dataclass(frozen=True)
class TokenFacts:
    """Numeric model and control-token facts owned by the package."""

    values: Mapping[str, int | tuple[int, ...]] = dataclasses.field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in self.values.items()
        }


@dataclasses.dataclass(frozen=True)
class TokenizerArtifact:
    """One package-relative tokenizer artifact."""

    location: str

    def to_metadata(self) -> dict[str, Any]:
        return {"location": self.location}


@dataclasses.dataclass(frozen=True)
class TokenizerFacts:
    """Tokenizer facts and package-relative artifacts.

    Mirrors onnx-genai's ``TokenizerFacts``. ``algorithm`` and ``vocab_size``
    are omitted when no tokenizer definition is available, while numeric token
    facts can still describe the package's execution contract.
    """

    algorithm: str | None = None
    vocab_size: int | None = None
    byte_level: bool = False
    artifacts: tuple[TokenizerArtifact, ...] = ()
    special_tokens: TokenFacts = dataclasses.field(default_factory=TokenFacts)

    def to_metadata(self) -> dict[str, Any]:
        facts: dict[str, Any] = {"byte_level": self.byte_level}
        if self.algorithm is not None:
            facts["algorithm"] = self.algorithm
        if self.vocab_size is not None:
            facts["vocab_size"] = self.vocab_size
        if self.artifacts:
            facts["artifacts"] = [artifact.to_metadata() for artifact in self.artifacts]
        if self.special_tokens.values:
            facts["special_tokens"] = self.special_tokens.to_metadata()
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
    """Whether any component addresses raw bytes rather than Unicode scalars.

    Only a ``ByteLevel`` stage qualifies.  ``ByteFallback`` does not: a
    SentencePiece tokenizer works in Unicode scalars with ``\u2581`` word marks
    and reaches for ``<0x..>`` pieces only when a character is absent from its
    vocabulary.  Calling that byte-level would tell a front end to apply the
    byte-to-Unicode mapping to text that never went through it.
    """
    if isinstance(node, dict):
        if node.get("type") == "ByteLevel":
            return True
        return any(_uses_byte_level(value) for value in node.values())
    if isinstance(node, list):
        return any(_uses_byte_level(value) for value in node)
    return False


def _byte_level_alphabet() -> frozenset[str]:
    """The 256 characters a byte-level vocabulary spells raw bytes with.

    Byte-level BPE maps every byte to one printable character so the vocabulary
    stays valid UTF-8.  A vocabulary that contains all 256 of them as
    single-character entries addresses bytes; one that does not, does not.  This
    is measurable evidence, which matters for legacy artifacts that ship no
    normalizer chain to state it.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    mapped = list(printable)
    spare = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            mapped.append(256 + spare)
            spare += 1
    return frozenset(chr(code) for code in mapped)


_BYTE_LEVEL_ALPHABET: Final = _byte_level_alphabet()


def _spells_raw_bytes(surfaces: Mapping[int, str]) -> bool:
    """Whether the vocabulary carries the whole byte-level alphabet."""
    return {form for form in surfaces.values() if len(form) == 1} >= _BYTE_LEVEL_ALPHABET


def _merge_added_tokens(surfaces: dict[int, str], added_tokens: Any) -> None:
    """Merge one added-token block, in any of the three shapes HF ships it in.

    ``tokenizer.json`` carries ``[{"id": .., "content": ..}, ..]``,
    ``tokenizer_config.json`` carries ``added_tokens_decoder`` keyed by id, and
    the legacy ``added_tokens.json`` is a plain ``content -> id`` table.  A
    checkpoint may declare a token in one and not the others: Qwen2-VL's
    ``<|image_pad|>`` is absent from its ``tokenizer.json`` and present only in
    ``added_tokens_decoder``, so reading a single block would drop exactly the
    placeholder a multimodal package exists to declare.
    """
    if isinstance(added_tokens, list):
        for added in added_tokens:
            if not isinstance(added, dict):
                continue
            token_id, content = added.get("id"), added.get("content")
            if isinstance(token_id, int) and not isinstance(token_id, bool) and content:
                surfaces[token_id] = str(content)
        return
    if not isinstance(added_tokens, dict):
        return
    for key, value in added_tokens.items():
        if isinstance(value, dict):
            # ``added_tokens_decoder``: id -> descriptor.
            content = value.get("content")
            if not content:
                continue
            try:
                token_id = int(key)
            except (TypeError, ValueError):
                continue
            if token_id >= 0:
                surfaces[token_id] = str(content)
        elif isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            # ``added_tokens.json``: content -> id.
            surfaces[value] = str(key)


def _surface_forms(vocab: Any, *added_blocks: Any) -> dict[int, str]:
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
    for block in added_blocks:
        _merge_added_tokens(surfaces, block)
    return surfaces


def _side_added_tokens(source: str) -> tuple[Any, ...]:
    """Added-token blocks that live outside ``tokenizer.json``."""
    blocks: list[Any] = []
    config_path = _source_asset_path(source, "tokenizer_config.json")
    if config_path is not None:
        config = _load_json(config_path)
        if isinstance(config, dict):
            blocks.append(config.get("added_tokens_decoder"))
    added_path = _source_asset_path(source, "added_tokens.json")
    if added_path is not None:
        blocks.append(_load_json(added_path))
    return tuple(blocks)


def _model_algorithm(model: Mapping[str, Any]) -> str | None:
    """Name the algorithm a ``tokenizer.json`` model body describes.

    ``model.type`` was only added to the format later, so the canonical
    WordPiece, Unigram and older BPE checkpoints on the Hub carry a model body
    with no type at all.  Each algorithm still leaves a distinct fingerprint in
    that body, so it is read from the fields present rather than defaulted.
    """
    declared = _ALGORITHMS.get(str(model.get("type", "")))
    if declared is not None:
        return declared
    if isinstance(model.get("merges"), list):
        return "bpe"
    if "continuing_subword_prefix" in model or "max_input_chars_per_word" in model:
        return "wordpiece"
    if "unk_id" in model or isinstance(model.get("vocab"), list):
        return "unigram"
    if isinstance(model.get("vocab"), dict):
        return "word_level"
    return None


def read_tokenizer_definition(source: str | None) -> TokenizerDefinition | None:
    """Read the packaged tokenizer's own vocabulary description.

    ``tokenizer.json`` is a complete tokenizer definition and is preferred: it
    states the whole vocabulary and the normalizer/pre-tokenizer chain that
    decides whether ids address bytes or characters.  When it parses, it is the
    answer — a legacy side file sitting next to it is a partial view of the same
    tokenizer and would contradict it.

    A package whose tokenizer predates that format ships the artifacts instead:
    a flat ``vocab.json``, plus ``merges.txt`` when and only when the vocabulary
    is a merge table.  That pair is itself the evidence for the algorithm -- a
    vocabulary with merges is BPE, one without is a flat word-level table where
    each entry is one unit -- and byte addressing is read from whether the
    vocabulary carries the byte-level alphabet.

    Either way, added tokens declared outside the definition are merged in: a
    checkpoint may name a special token only in ``tokenizer_config.json`` or
    ``added_tokens.json``, and dropping it would leave the role unstatable.

    Returns ``None`` when no tokenizer definition is reachable.  A package that
    cannot show its vocabulary states no tokenizer facts at all, because a
    plausible-looking guess would be indistinguishable from a measured value.
    """
    if not source:
        return None
    side_blocks = _side_added_tokens(source)

    path = _source_asset_path(source, "tokenizer.json")
    if path is not None:
        definition = _load_json(path)
        if isinstance(definition, dict):
            model = definition.get("model")
            model = model if isinstance(model, dict) else {}
            algorithm = _model_algorithm(model)
            surfaces = _surface_forms(
                model.get("vocab"), definition.get("added_tokens"), *side_blocks
            )
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
    surfaces = _surface_forms(vocab, *side_blocks)
    if not surfaces:
        return None
    has_merges = _source_asset_path(source, "merges.txt") is not None
    return TokenizerDefinition(
        algorithm="bpe" if has_merges else "word_level",
        byte_level=_spells_raw_bytes(surfaces),
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
    """Coerce one declared token id."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _token_ids(value: Any) -> tuple[int, ...]:
    """Coerce an ordered token-id set without silently dropping multi-EOS."""
    values = value if isinstance(value, (list, tuple)) else (value,)
    result: list[int] = []
    for candidate in values:
        token_id = _token_id(candidate)
        if token_id is not None and token_id not in result:
            result.append(token_id)
    return tuple(result)


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


def build_token_facts(
    config: Any,
    roles: Iterable[SpecialTokenRole],
    *,
    declared: Mapping[str, Any] | None = None,
) -> TokenFacts:
    """Resolve numeric package token facts.

    ``declared`` overrides the config for roles the emitting workflow has
    already resolved for itself, so a document never states one id for its
    termination policy and a different one for the same role here.

    Multi-valued roles preserve their declared order. Scalar roles take exactly
    one non-negative integer. Token spellings remain in tokenizer assets.
    """
    declared = declared or {}
    values: dict[str, int | tuple[int, ...]] = {}
    for role in roles:
        value = declared.get(role.name)
        if value is None:
            for field in role.fields:
                value = _config_value(config, field)
                if value is not None:
                    break
        if role.multiple:
            token_ids = _token_ids(value)
            if token_ids:
                values[role.name] = token_ids
            continue
        token_id = _token_id(value)
        if token_id is not None:
            values[role.name] = token_id
    return TokenFacts(values)


def build_tokenizer_facts(
    source: str | None,
    config: Any,
    *,
    roles: Sequence[SpecialTokenRole] = TEXT_TOKEN_ROLES,
    package_dir: str | None = None,
) -> TokenizerFacts | None:
    """Describe the packaged tokenizer, or ``None`` when it cannot be read.

    Args:
        source: HuggingFace model id or local directory carrying the source
            checkpoint's tokenizer artifacts.
        config: Resolved architecture config; supplies each role's id.
        roles: Semantic roles this package can state.  A text package states
            text roles; a multimodal one adds the media placeholders whose
            positions its encoder features replace.
        package_dir: The materialized package, when it exists.  Its tokenizer
            outranks the source's: the writers rebuild it through the fast
            backend, which folds every added token into one definition, whereas
            a source checkpoint may scatter them across side files it does not
            ship.  It is also the only thing that can name package-relative
            artifacts, since only the package says which files it contains.

    Returns:
        The facts, or ``None`` when no tokenizer definition is reachable.

    ``vocab_size`` is the tokenizer's own vocabulary width rather than the
    decoder's logits width.  The two usually agree; where they differ the logits
    tensor is padded to a hardware-friendly multiple and the extra columns
    address nothing, so the vocabulary is the width a caller can actually render.
    """
    definition = read_tokenizer_definition(package_dir) or read_tokenizer_definition(source)
    special_tokens = build_token_facts(
        config,
        roles,
        declared=source_declared_roles(source, roles),
    )
    artifacts = ()
    if package_dir is not None:
        artifacts = tuple(
            TokenizerArtifact(location=name)
            for name in TOKENIZER_ARTIFACT_NAMES
            if os.path.isfile(os.path.join(package_dir, name))
        )
    if definition is None and not artifacts and not special_tokens.values:
        return None
    return TokenizerFacts(
        algorithm=definition.algorithm if definition is not None else None,
        vocab_size=definition.vocab_size if definition is not None else None,
        byte_level=definition.byte_level if definition is not None else False,
        artifacts=artifacts,
        special_tokens=special_tokens,
    )


def attach_package_facts(
    metadata: dict[str, Any],
    source: str | None,
    config: Any,
    *,
    roles: Sequence[SpecialTokenRole] = TEXT_TOKEN_ROLES,
    package_dir: str | None = None,
) -> None:
    """Publish tokenizer facts under ``metadata['package']`` when they are known.

    Builders call this with the source checkpoint alone; writers call it again
    with the materialized package, which outranks it and adds the artifact
    locations.  The section is left absent when nothing can be read, so a reader
    can tell "this package does not state its tokenizer" apart from "this
    package states an empty tokenizer".
    """
    tokenizer = build_tokenizer_facts(source, config, roles=roles, package_dir=package_dir)
    if tokenizer is None:
        return
    metadata["schema_version"] = "v1.2"
    metadata.setdefault("package", {})["tokenizer"] = tokenizer.to_metadata()
