# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for upstream asset correction."""

from __future__ import annotations

import hashlib
import json
import pathlib
import tempfile
import unittest

from mobius.upstream_patches import apply_asset_patches, available_patches

_ORIGINAL = "hello {{ name }}\ngoodbye\n"
_PATCHED = "hello {{ name }}\nfarewell\n"
_DIFF = "@@ -1,2 +1,2 @@\n hello {{ name }}\n-goodbye\n+farewell\n"


def _write_patch_root(root: pathlib.Path, *, operations: list[dict]) -> pathlib.Path:
    directory = root / "owner" / "model"
    directory.mkdir(parents=True)
    (directory / "chat_template.jinja.patch").write_text(_DIFF, encoding="utf-8")
    (directory / "patch.json").write_text(
        json.dumps(
            {
                "upstream": {"repo": "owner/model", "revision": "0" * 40},
                "summary": "say farewell",
                "operations": operations,
            }
        ),
        encoding="utf-8",
    )
    return root


def _template_operation() -> dict:
    return {
        "kind": "diff",
        "target": "chat_template.jinja",
        "patch": "chat_template.jinja.patch",
        "upstream_sha256": hashlib.sha256(_ORIGINAL.encode()).hexdigest(),
    }


def _sync_operation() -> dict:
    return {
        "kind": "sync_chat_template",
        "target": "tokenizer_config.json",
        "source": "chat_template.jinja",
    }


class ApplyAssetPatchesTest(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._temp.name)
        self.package = self.root / "package"
        self.package.mkdir()
        self.patches = _write_patch_root(
            self.root / "patches",
            operations=[_template_operation(), _sync_operation()],
        )
        self.addCleanup(self._temp.cleanup)

    def _write_package(self, template: str = _ORIGINAL, *, tokenizer_config=True):
        (self.package / "chat_template.jinja").write_text(template, encoding="utf-8")
        if tokenizer_config:
            (self.package / "tokenizer_config.json").write_text(
                json.dumps({"chat_template": template, "model_max_length": 4096}),
                encoding="utf-8",
            )

    def _apply(self):
        return apply_asset_patches(self.package, root=self.patches)

    def test_corrects_a_matching_asset(self):
        self._write_package()
        self.assertEqual(self._apply(), ["chat_template.jinja", "tokenizer_config.json"])
        self.assertEqual(
            (self.package / "chat_template.jinja").read_text(encoding="utf-8"),
            _PATCHED,
        )

    def test_syncs_the_template_duplicated_into_the_tokenizer_config(self):
        self._write_package()
        self._apply()
        with (self.package / "tokenizer_config.json").open(encoding="utf-8") as handle:
            config = json.load(handle)
        self.assertEqual(config["chat_template"], _PATCHED)
        self.assertEqual(config["model_max_length"], 4096, "unrelated keys survive")

    def test_leaves_an_asset_upstream_already_fixed(self):
        self._write_package(template=_PATCHED)
        self.assertEqual(self._apply(), [])
        self.assertEqual(
            (self.package / "chat_template.jinja").read_text(encoding="utf-8"),
            _PATCHED,
        )

    def test_is_idempotent(self):
        self._write_package()
        self.assertTrue(self._apply())
        self.assertEqual(self._apply(), [], "a corrected package is left alone")

    def test_skips_the_sync_when_the_template_was_not_corrected(self):
        self._write_package(template=_PATCHED)
        self._apply()
        with (self.package / "tokenizer_config.json").open(encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["chat_template"], _PATCHED)

    def test_tolerates_a_package_missing_the_target(self):
        self.assertEqual(self._apply(), [])

    def test_tolerates_an_unreadable_tokenizer_config(self):
        self._write_package()
        (self.package / "tokenizer_config.json").write_text("{not json", encoding="utf-8")
        with self.assertLogs("mobius.upstream_patches._patches", level="ERROR"):
            applied = self._apply()
        self.assertEqual(
            applied,
            ["chat_template.jinja"],
            "the template is still corrected when its duplicate cannot be read",
        )
        self.assertEqual(
            (self.package / "tokenizer_config.json").read_text(encoding="utf-8"),
            "{not json",
            "an unreadable config is left exactly as found, not truncated",
        )

    def test_tolerates_a_missing_package_directory(self):
        self.assertEqual(apply_asset_patches(self.package / "absent", root=self.patches), [])


class ShippedPatchesTest(unittest.TestCase):
    def test_every_shipped_patch_is_well_formed(self):
        patches = available_patches()
        self.assertTrue(patches, "the package ships at least one correction")
        for patch in patches:
            with self.subTest(patch=patch.name):
                self.assertTrue(patch.summary, "a patch explains itself")
                self.assertEqual(len(patch.upstream_revision), 40, "pinned by full sha")
                self.assertTrue(
                    (patch.directory / "README.md").is_file(),
                    "a patch records why it exists",
                )
                for operation in patch.operations:
                    self.assertIn(operation["kind"], {"diff", "sync_chat_template"})
                    if operation["kind"] == "diff":
                        self.assertTrue((patch.directory / operation["patch"]).is_file())
                        self.assertEqual(len(operation["upstream_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
