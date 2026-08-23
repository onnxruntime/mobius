# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for canonical ``package.tokenizer`` facts emission.

A published package carries graphs and a workflow, but a request arrives as text
and media.  These tests pin the facts that turn one into the other: which
vocabulary the emitted ids belong to, and — for a multimodal package — which
prompt token an image's features replace.  Without the latter the package
declares no place in the token stream for its own image features, so an attached
image is preprocessed and then has nowhere to go.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import jsonschema
import pytest
import yaml

from mobius._configs import MMSConfig
from mobius._model_package import ModelPackage
from mobius.integrations.onnx_genai.auto_export import write_onnx_genai_config
from mobius.integrations.onnx_genai.auto_export_test import _vlm_package, _VlmCfg
from mobius.integrations.onnx_genai.inference_metadata_test import (
    _decoder_model,
    _onnx_genai_schema_path,
)
from mobius.integrations.onnx_genai.package_facts import (
    IMAGE_PLACEHOLDER_ROLE,
    _byte_level_alphabet,
    read_tokenizer_definition,
)
from mobius.integrations.onnx_genai.workflow_metadata import (
    build_ctc_asr_workflow_metadata,
    build_decoder_workflow_metadata,
    build_speculative_workflow_metadata,
    build_vlm_workflow_metadata,
    write_ctc_asr_workflow_metadata,
    write_decoder_workflow_metadata,
)
from mobius.integrations.onnx_genai.workflow_metadata_test import _speculative_package
from mobius.models.wav2vec2_ctc import Wav2Vec2ForCTCModel
from mobius.tasks._ctc_asr import CTCAsrTask

# Ids the fixture tokenizer assigns to its added special tokens.  They are
# deliberately above the base vocabulary, the way a real checkpoint appends
# them, so ``vocab_size`` cannot be recovered from the base table alone.
_BOS_ID = 100
_EOS_ID = 101
_PAD_ID = 102
_IMAGE_ID = 103
_VIDEO_ID = 104
_TEXT_VOCAB_SIZE = _PAD_ID + 1
_MEDIA_VOCAB_SIZE = _VIDEO_ID + 1
# The decoder fixture's logits are wider than the vocabulary, the way a
# checkpoint padded to a hardware-friendly multiple is.
_LOGITS_WIDTH = 128


@dataclasses.dataclass
class _TextCfg:
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 64
    hidden_size: int = 1024
    max_position_embeddings: int = 8192
    sliding_window: int | None = None
    model_type: str = "qwen"
    vocab_size: int = _LOGITS_WIDTH
    bos_token_id: int = _BOS_ID
    eos_token_id: int = _EOS_ID
    pad_token_id: int = _PAD_ID


@dataclasses.dataclass
class _MediaCfg(_VlmCfg):
    """A vision config that can also declare a video placeholder."""

    video_token_id: int | None = None


def _write_tokenizer(
    directory: Path,
    *,
    byte_level: bool = True,
    added: dict[int, str] | None = None,
) -> Path:
    """Write a minimal ``tokenizer.json`` with an appended special-token block."""
    added = {
        _BOS_ID: "<s>",
        _EOS_ID: "</s>",
        _PAD_ID: "<pad>",
        **(added or {}),
    }
    tokenizer = {
        "model": {
            "type": "BPE",
            "vocab": {
                chr(ord("a") + index % 26) * (1 + index // 26): index for index in range(100)
            },
            "merges": [],
        },
        "added_tokens": [
            {"id": token_id, "content": content} for token_id, content in added.items()
        ],
    }
    if byte_level:
        tokenizer["pre_tokenizer"] = {"type": "ByteLevel", "add_prefix_space": False}
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "tokenizer.json"
    path.write_text(json.dumps(tokenizer), encoding="utf-8")
    return path


def _decoder_package(config: _TextCfg) -> ModelPackage:
    model = _decoder_model([], position_shape=["batch", "sequence"], raw_token_input=True)
    return ModelPackage({"model": model}, config=config)


def _validate(metadata: dict) -> None:
    with open(_onnx_genai_schema_path(), encoding="utf-8") as handle:
        jsonschema.validate(instance=metadata, schema=json.load(handle))


class TestTokenizerDefinition:
    """The vocabulary is read from the package's own artifacts, never guessed."""

    def test_reads_algorithm_vocabulary_and_byte_level_from_tokenizer_json(self, tmp_path):
        _write_tokenizer(tmp_path)
        definition = read_tokenizer_definition(str(tmp_path))
        assert definition is not None
        assert definition.algorithm == "bpe"
        assert definition.byte_level is True
        # Added tokens sit above the base table, so the width a caller must
        # assume is the highest occupied id plus one, not the entry count.
        assert definition.vocab_size == _TEXT_VOCAB_SIZE
        assert definition.surface_forms[_EOS_ID] == "</s>"

    def test_a_flat_vocabulary_without_merges_is_word_level(self, tmp_path):
        (tmp_path / "vocab.json").write_text(
            json.dumps({"<pad>": 0, "|": 1, "A": 2}), encoding="utf-8"
        )
        definition = read_tokenizer_definition(str(tmp_path))
        assert definition is not None
        assert definition.algorithm == "word_level"
        assert definition.byte_level is False
        assert definition.vocab_size == 3

    def test_a_vocabulary_with_merges_is_bpe(self, tmp_path):
        (tmp_path / "vocab.json").write_text(json.dumps({"a": 0, "b": 1}), encoding="utf-8")
        (tmp_path / "merges.txt").write_text("a b\n", encoding="utf-8")
        definition = read_tokenizer_definition(str(tmp_path))
        assert definition is not None
        assert definition.algorithm == "bpe"

    def test_an_unreadable_tokenizer_states_nothing(self, tmp_path):
        assert read_tokenizer_definition(str(tmp_path)) is None
        assert read_tokenizer_definition(None) is None


class TestDecoderPackageFacts:
    """A text package states the vocabulary the ids it emits belong to."""

    def test_facts_survive_into_package_metadata(self, tmp_path):
        _write_tokenizer(tmp_path)
        metadata = build_decoder_workflow_metadata(
            _decoder_package(_TextCfg()), _TextCfg(), source=str(tmp_path)
        )
        tokenizer = metadata["package"]["tokenizer"]
        assert tokenizer["algorithm"] == "bpe"
        assert tokenizer["vocab_size"] == _TEXT_VOCAB_SIZE
        assert tokenizer["byte_level"] is True
        assert tokenizer["special_tokens"] == {
            "bos": {"id": _BOS_ID, "content": "<s>"},
            "eos": {"id": _EOS_ID, "content": "</s>"},
            "pad": {"id": _PAD_ID, "content": "<pad>"},
        }
        _validate(metadata)

    def test_the_stated_width_is_the_vocabulary_not_the_logits_row(self, tmp_path):
        # Logits are often padded to a hardware-friendly multiple; those extra
        # columns address no token and render nothing. A caller that trusted
        # the row width would believe in ids the tokenizer cannot decode.
        _write_tokenizer(tmp_path)
        metadata = build_decoder_workflow_metadata(
            _decoder_package(_TextCfg()), _TextCfg(), source=str(tmp_path)
        )
        logits = metadata["pipeline"]["workflow"]["state"]["logits"]["contract"]
        assert logits["shape"][-1] == _LOGITS_WIDTH
        assert metadata["package"]["tokenizer"]["vocab_size"] == _TEXT_VOCAB_SIZE

    def test_pipeline_workflow_remains_the_sole_representation(self, tmp_path):
        _write_tokenizer(tmp_path)
        metadata = build_decoder_workflow_metadata(
            _decoder_package(_TextCfg()), _TextCfg(), source=str(tmp_path)
        )
        assert set(metadata["pipeline"]) == {"workflow"}
        assert "models" not in metadata["pipeline"]
        assert "model" not in metadata

    def test_the_stated_stop_token_is_the_one_the_loop_terminates_on(self, tmp_path):
        # A repackaged checkpoint may be retuned without its config being
        # rewritten, so genai_config.json outranks the config. Both the
        # termination policy and the published role must follow it, or one
        # document names two different stop tokens.
        _write_tokenizer(tmp_path, added={_IMAGE_ID: "<|endoftext|>"})
        (tmp_path / "genai_config.json").write_text(
            json.dumps({"model": {"eos_token_id": _IMAGE_ID}}), encoding="utf-8"
        )
        metadata = build_decoder_workflow_metadata(
            _decoder_package(_TextCfg()), _TextCfg(), source=str(tmp_path)
        )
        workflow = metadata["pipeline"]["workflow"]
        assert metadata["package"]["tokenizer"]["special_tokens"]["eos"] == {
            "id": _IMAGE_ID,
            "content": "<|endoftext|>",
        }
        assert workflow["inputs"]["package.eos_ids"]["default"] == _IMAGE_ID

    def test_a_role_the_vocabulary_cannot_render_is_omitted(self, tmp_path):
        # A config field pointing outside the shipped vocabulary is stale. A
        # front end that spliced that id would emit a token the tokenizer
        # cannot render, so the role is dropped rather than half-stated.
        _write_tokenizer(tmp_path)
        metadata = build_decoder_workflow_metadata(
            _decoder_package(_TextCfg()),
            _TextCfg(bos_token_id=9_999),
            source=str(tmp_path),
        )
        assert "bos" not in metadata["package"]["tokenizer"]["special_tokens"]

    def test_a_package_without_a_readable_tokenizer_states_no_facts(self, tmp_path):
        metadata = build_decoder_workflow_metadata(
            _decoder_package(_TextCfg()), _TextCfg(), source=str(tmp_path)
        )
        assert "package" not in metadata

    def test_artifacts_name_only_files_the_package_contains(self, tmp_path):
        source = tmp_path / "source"
        _write_tokenizer(source)
        (source / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        # ``merges.txt`` exists in the source but is never copied, so naming it
        # would hand a reader a package-relative path that does not resolve.
        (source / "merges.txt").write_text("", encoding="utf-8")
        package_dir = tmp_path / "package"
        _write_tokenizer(package_dir)

        write_decoder_workflow_metadata(
            _decoder_package(_TextCfg()),
            str(package_dir),
            _TextCfg(),
            source=str(source),
        )
        metadata = yaml.safe_load((package_dir / "inference_metadata.yaml").read_text())
        assert metadata["package"]["tokenizer"]["artifacts"] == [
            {"location": "tokenizer.json"}
        ]

    def test_the_dispatcher_ships_assets_before_the_document_names_them(self, tmp_path):
        source = tmp_path / "source"
        _write_tokenizer(source)
        (source / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        package_dir = tmp_path / "package"

        artifacts = write_onnx_genai_config(
            _decoder_package(_TextCfg()),
            str(package_dir),
            config=_TextCfg(),
            source=str(source),
        )
        metadata = yaml.safe_load(Path(artifacts["inference_metadata"]).read_text())
        locations = [
            entry["location"] for entry in metadata["package"]["tokenizer"]["artifacts"]
        ]
        assert locations == ["tokenizer.json", "tokenizer_config.json"]
        assert all((package_dir / location).is_file() for location in locations)


class TestMultimodalPackageFacts:
    """A multimodal package must state where an image goes in the prompt."""

    @pytest.fixture
    def source(self, tmp_path):
        directory = tmp_path / "source"
        _write_tokenizer(
            directory,
            added={_IMAGE_ID: "<image>", _VIDEO_ID: "<video>"},
        )
        return directory

    def _metadata(self, source, **overrides):
        config = _MediaCfg(image_token_id=_IMAGE_ID, eos_token_id=_EOS_ID, **overrides)
        return build_vlm_workflow_metadata(_vlm_package(), config, source=str(source))

    def test_image_placeholder_role_is_published(self, source):
        tokenizer = self._metadata(source)["package"]["tokenizer"]
        assert tokenizer["special_tokens"][IMAGE_PLACEHOLDER_ROLE] == {
            "id": _IMAGE_ID,
            "content": "<image>",
        }
        assert tokenizer["special_tokens"]["eos"] == {"id": _EOS_ID, "content": "</s>"}

    def test_image_input_routing_survives_alongside_the_facts(self, source):
        metadata = self._metadata(source)
        workflow = metadata["pipeline"]["workflow"]
        # The encoded image still enters as a typed media request input, is
        # still preprocessed by the package's own program, and the program's
        # outputs still reach the vision encoder. Facts describe that route;
        # they must not replace it.
        media = workflow["inputs"]["request.image"]
        assert media["role"] == {"kind": "runtime", "version": "1.0", "role": "media"}
        assert media["contract"]["dtype"] == "uint8"
        assert media["contract"]["rank"] == 1
        preprocess = next(
            step
            for step in workflow["steps"][0]["setup"]
            if step["component"] == "image_preprocess"
        )
        assert preprocess["inputs"] == {"encoded": "request.image"}
        vision = next(
            step
            for step in workflow["steps"][0]["setup"]
            if step["component"] == "vision_encoder"
        )
        assert set(vision["inputs"].values()) <= set(preprocess["outputs"].values())
        assert {
            output["name"] for output in metadata["preprocessing"]["image"]["outputs"]
        } == set(preprocess["outputs"].values())
        _validate(metadata)

    def test_a_declared_video_placeholder_is_published_too(self, source):
        tokenizer = self._metadata(source, video_token_id=_VIDEO_ID)["package"]["tokenizer"]
        assert tokenizer["special_tokens"]["video_placeholder"] == {
            "id": _VIDEO_ID,
            "content": "<video>",
        }

    def test_an_undeclared_modality_role_is_absent(self, source):
        tokenizer = self._metadata(source)["package"]["tokenizer"]
        assert "video_placeholder" not in tokenizer["special_tokens"]
        assert "audio_placeholder" not in tokenizer["special_tokens"]


class TestScatteredAddedTokens:
    """A checkpoint may name a special token outside its tokenizer definition."""

    def test_a_token_declared_only_in_tokenizer_config_is_still_rendered(self, tmp_path):
        # Qwen2-VL's `<|image_pad|>` is absent from its own tokenizer.json and
        # present only in tokenizer_config.json's added_tokens_decoder. Reading
        # one block would drop exactly the placeholder a multimodal package
        # exists to declare, leaving the image nowhere to go.
        _write_tokenizer(tmp_path)
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps(
                {"added_tokens_decoder": {str(_IMAGE_ID): {"content": "<|image_pad|>"}}}
            ),
            encoding="utf-8",
        )
        definition = read_tokenizer_definition(str(tmp_path))
        assert definition is not None
        assert definition.surface_forms[_IMAGE_ID] == "<|image_pad|>"
        assert definition.vocab_size == _IMAGE_ID + 1

    def test_the_legacy_added_tokens_table_is_merged_too(self, tmp_path):
        _write_tokenizer(tmp_path)
        (tmp_path / "added_tokens.json").write_text(
            json.dumps({"<|video_pad|>": _VIDEO_ID}), encoding="utf-8"
        )
        definition = read_tokenizer_definition(str(tmp_path))
        assert definition is not None
        assert definition.surface_forms[_VIDEO_ID] == "<|video_pad|>"

    def test_the_multimodal_package_publishes_the_scattered_placeholder(self, tmp_path):
        _write_tokenizer(tmp_path)
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps(
                {"added_tokens_decoder": {str(_IMAGE_ID): {"content": "<|image_pad|>"}}}
            ),
            encoding="utf-8",
        )
        config = _MediaCfg(image_token_id=_IMAGE_ID, eos_token_id=_EOS_ID)
        metadata = build_vlm_workflow_metadata(_vlm_package(), config, source=str(tmp_path))
        assert metadata["package"]["tokenizer"]["special_tokens"][IMAGE_PLACEHOLDER_ROLE] == {
            "id": _IMAGE_ID,
            "content": "<|image_pad|>",
        }


class TestAlgorithmIsReadNotDefaulted:
    """`model.type` postdates the format, so the model body is the evidence."""

    @staticmethod
    def _write(directory: Path, model: dict, **sections) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tokenizer.json").write_text(
            json.dumps({"model": model, **sections}), encoding="utf-8"
        )

    def test_a_wordpiece_body_without_a_type_is_wordpiece(self, tmp_path):
        self._write(
            tmp_path,
            {
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100,
                "unk_token": "[UNK]",
                "vocab": {"[PAD]": 0, "the": 1, "##ing": 2},
            },
        )
        definition = read_tokenizer_definition(str(tmp_path))
        assert definition is not None
        assert definition.algorithm == "wordpiece"

    def test_a_unigram_body_without_a_type_is_unigram(self, tmp_path):
        self._write(
            tmp_path, {"unk_id": 2, "vocab": [["<pad>", 0.0], ["</s>", 0.0], ["a", -1.0]]}
        )
        definition = read_tokenizer_definition(str(tmp_path))
        assert definition is not None
        assert definition.algorithm == "unigram"
        assert definition.surface_forms[2] == "a"

    def test_a_typeless_definition_outranks_a_legacy_file_beside_it(self, tmp_path):
        # A definition that parses is the answer. Falling through to the flat
        # side file would report byte_level=False about a tokenizer whose own
        # chain says otherwise -- two contradictory claims from one checkpoint.
        self._write(
            tmp_path,
            {"merges": [], "vocab": {"a": 0, "b": 1}},
            pre_tokenizer={"type": "ByteLevel", "add_prefix_space": False},
        )
        (tmp_path / "vocab.json").write_text(json.dumps({"a": 0}), encoding="utf-8")
        definition = read_tokenizer_definition(str(tmp_path))
        assert definition is not None
        assert definition.algorithm == "bpe"
        assert definition.byte_level is True

    def test_byte_addressing_is_measured_from_a_legacy_vocabulary(self, tmp_path):
        # Legacy artifacts carry no normalizer chain, but a byte-level
        # vocabulary still shows it: every one of the 256 byte glyphs is a
        # single-character entry.
        alphabet = sorted(_byte_level_alphabet())
        vocab = {glyph: index for index, glyph in enumerate(alphabet)}
        vocab["\u0120the"] = len(vocab)
        (tmp_path / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
        (tmp_path / "merges.txt").write_text("\u0120 t\n", encoding="utf-8")
        definition = read_tokenizer_definition(str(tmp_path))
        assert definition is not None
        assert definition.algorithm == "bpe"
        assert definition.byte_level is True

    def test_a_character_vocabulary_is_not_called_byte_level(self, tmp_path):
        (tmp_path / "vocab.json").write_text(
            json.dumps({"<pad>": 0, "|": 1, "A": 2}), encoding="utf-8"
        )
        definition = read_tokenizer_definition(str(tmp_path))
        assert definition is not None
        assert definition.byte_level is False


class TestPackageOutranksSource:
    """The materialized package is the tokenizer a reader will actually load."""

    def test_the_shipped_definition_wins_over_the_source_checkpoint(self, tmp_path):
        # Writers rebuild tokenizer.json through the fast backend, folding every
        # scattered added token into one definition. Reporting the source's
        # partial view instead would contradict the file shipped beside it.
        source = tmp_path / "source"
        _write_tokenizer(source)
        package_dir = tmp_path / "package"
        _write_tokenizer(package_dir, added={_IMAGE_ID: "<image>", _VIDEO_ID: "<video>"})

        write_decoder_workflow_metadata(
            _decoder_package(_TextCfg()), str(package_dir), _TextCfg(), source=str(source)
        )
        metadata = yaml.safe_load((package_dir / "inference_metadata.yaml").read_text())
        assert metadata["package"]["tokenizer"]["vocab_size"] == _MEDIA_VOCAB_SIZE

    def test_every_tokenizer_asset_the_writers_copy_can_be_declared(self, tmp_path):
        source = tmp_path / "source"
        _write_tokenizer(source)
        package_dir = tmp_path / "package"
        _write_tokenizer(package_dir)
        for name in ("tokenizer.model", "added_tokens.json", "chat_template.jinja"):
            (package_dir / name).write_text("x", encoding="utf-8")

        write_decoder_workflow_metadata(
            _decoder_package(_TextCfg()), str(package_dir), _TextCfg(), source=str(source)
        )
        metadata = yaml.safe_load((package_dir / "inference_metadata.yaml").read_text())
        assert [
            entry["location"] for entry in metadata["package"]["tokenizer"]["artifacts"]
        ] == ["tokenizer.json", "tokenizer.model", "added_tokens.json", "chat_template.jinja"]


class TestCtcPackageFacts:
    """A CTC package renders a transcript from class ids, so it has a vocabulary."""

    @staticmethod
    def _config(**overrides) -> MMSConfig:
        base = {
            "vocab_size": 32,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "conv_dim": (32, 32, 64),
            "conv_kernel": (10, 3, 2),
            "conv_stride": (5, 2, 2),
            "conv_bias": False,
            "feat_extract_norm": "layer",
            "do_stable_layer_norm": True,
            "pad_token_id": 0,
        }
        base.update(overrides)
        return MMSConfig(**base)

    @staticmethod
    def _write_vocabulary(directory: Path, size: int) -> None:
        table = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4}
        table.update({chr(ord("A") + index): 5 + index for index in range(size - 5)})
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "vocab.json").write_text(json.dumps(table), encoding="utf-8")

    def test_blank_id_and_the_published_pad_role_name_one_token(self, tmp_path):
        config = self._config()
        self._write_vocabulary(tmp_path, config.vocab_size)
        metadata = build_ctc_asr_workflow_metadata(
            CTCAsrTask().build(Wav2Vec2ForCTCModel(config), config),
            config,
            source=str(tmp_path),
        )
        tokenizer = metadata["package"]["tokenizer"]
        assert tokenizer["algorithm"] == "word_level"
        assert tokenizer["vocab_size"] == config.vocab_size
        blank_id = metadata["profiles"]["transcription"]["decoding"]["blank_id"]
        assert tokenizer["special_tokens"]["pad"] == {"id": blank_id, "content": "<pad>"}
        _validate(metadata)

    def test_a_package_that_ships_no_vocabulary_still_states_its_facts(self, tmp_path):
        # `auto_export`'s CTC branch reaches for a fast tokenizer, which a
        # Wav2Vec2/MMS checkpoint does not have, so the package ends up with no
        # vocabulary file. The facts are self-contained and survive that; the
        # artifact list must not name a file the package does not contain.
        config = self._config()
        source = tmp_path / "source"
        self._write_vocabulary(source, config.vocab_size)
        package_dir = tmp_path / "package"

        write_ctc_asr_workflow_metadata(
            CTCAsrTask().build(Wav2Vec2ForCTCModel(config), config),
            str(package_dir),
            config,
            source=str(source),
        )
        metadata = yaml.safe_load((package_dir / "inference_metadata.yaml").read_text())
        tokenizer = metadata["package"]["tokenizer"]
        assert tokenizer["vocab_size"] == config.vocab_size
        assert "artifacts" not in tokenizer

    def test_the_writer_declares_the_vocabulary_it_shipped(self, tmp_path):
        config = self._config()
        source = tmp_path / "source"
        self._write_vocabulary(source, config.vocab_size)
        package_dir = tmp_path / "package"
        package_dir.mkdir()
        self._write_vocabulary(package_dir, config.vocab_size)

        write_ctc_asr_workflow_metadata(
            CTCAsrTask().build(Wav2Vec2ForCTCModel(config), config),
            str(package_dir),
            config,
            source=str(source),
        )
        metadata = yaml.safe_load((package_dir / "inference_metadata.yaml").read_text())
        assert metadata["package"]["tokenizer"]["artifacts"] == [{"location": "vocab.json"}]


class TestSpeculativePackageFacts:
    """Proposer and verifier are two views of one vocabulary."""

    def test_facts_are_stated_once_for_both_components(self, tmp_path):
        _write_tokenizer(tmp_path)
        metadata = build_speculative_workflow_metadata(
            _speculative_package(), _TextCfg(), source=str(tmp_path)
        )
        tokenizer = metadata["package"]["tokenizer"]
        assert tokenizer["vocab_size"] == _TEXT_VOCAB_SIZE
        assert tokenizer["special_tokens"]["eos"] == {"id": _EOS_ID, "content": "</s>"}
        # Nothing was shipped next to the document, so it claims no artifacts.
        assert "artifacts" not in tokenizer
        assert set(metadata["pipeline"]) == {"workflow"}
        _validate(metadata)
