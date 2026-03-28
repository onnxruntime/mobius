# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the feature flag system."""

from __future__ import annotations

import pytest

from mobius._flags import Flags, flags, list_flags, override_flags


class TestDefaultValues:
    """Flags have the expected defaults when no env vars are set."""

    def test_mmap_loading_default_off(self, monkeypatch):
        monkeypatch.delenv("MOBIUS_MMAP_LOADING", raising=False)
        f = Flags()
        assert f.mmap_loading is False

    def test_lazy_weight_loading_default_on(self, monkeypatch):
        monkeypatch.delenv("MOBIUS_LAZY_WEIGHT_LOADING", raising=False)
        f = Flags()
        assert f.lazy_weight_loading is True

    def test_suppress_dedup_warning_default_on(self, monkeypatch):
        monkeypatch.delenv("MOBIUS_SUPPRESS_DEDUP_WARNING", raising=False)
        f = Flags()
        assert f.suppress_dedup_warning is True


class TestEnvVarOverride:
    """Flags read from environment variables at construction time."""

    @pytest.mark.parametrize("val", ["1", "true", "True", "TRUE", "yes", "YES"])
    def test_truthy_values(self, monkeypatch, val):
        monkeypatch.setenv("MOBIUS_MMAP_LOADING", val)
        f = Flags()
        assert f.mmap_loading is True

    @pytest.mark.parametrize("val", ["0", "false", "False", "FALSE", "no", "NO"])
    def test_falsy_values(self, monkeypatch, val):
        monkeypatch.setenv("MOBIUS_LAZY_WEIGHT_LOADING", val)
        f = Flags()
        assert f.lazy_weight_loading is False

    def test_unknown_value_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MOBIUS_MMAP_LOADING", "maybe")
        f = Flags()
        assert f.mmap_loading is False  # default

    def test_suppress_dedup_warning_env_var(self, monkeypatch):
        monkeypatch.setenv("MOBIUS_SUPPRESS_DEDUP_WARNING", "0")
        f = Flags()
        assert f.suppress_dedup_warning is False


class TestProgrammaticOverride:
    """Flags can be assigned directly on the singleton."""

    def test_set_mmap_loading(self):
        original = flags.mmap_loading
        try:
            flags.mmap_loading = not original
            assert flags.mmap_loading is not original
        finally:
            flags.mmap_loading = original

    def test_set_lazy_weight_loading(self):
        original = flags.lazy_weight_loading
        try:
            flags.lazy_weight_loading = not original
            assert flags.lazy_weight_loading is not original
        finally:
            flags.lazy_weight_loading = original

    def test_set_suppress_dedup_warning(self):
        original = flags.suppress_dedup_warning
        try:
            flags.suppress_dedup_warning = not original
            assert flags.suppress_dedup_warning is not original
        finally:
            flags.suppress_dedup_warning = original


class TestOverrideFlagsContextManager:
    """override_flags() restores original values on exit."""

    def test_single_flag_restored(self):
        original = flags.mmap_loading
        with override_flags(mmap_loading=not original):
            assert flags.mmap_loading is not original
        assert flags.mmap_loading is original

    def test_multiple_flags_restored(self):
        orig_mmap = flags.mmap_loading
        orig_lazy = flags.lazy_weight_loading
        with override_flags(mmap_loading=not orig_mmap, lazy_weight_loading=not orig_lazy):
            assert flags.mmap_loading is not orig_mmap
            assert flags.lazy_weight_loading is not orig_lazy
        assert flags.mmap_loading is orig_mmap
        assert flags.lazy_weight_loading is orig_lazy

    def test_restored_on_exception(self):
        original = flags.mmap_loading
        with pytest.raises(RuntimeError), override_flags(mmap_loading=not original):
            raise RuntimeError("boom")
        assert flags.mmap_loading is original

    def test_nested_overrides(self):
        original = flags.mmap_loading
        with override_flags(mmap_loading=not original):
            with override_flags(mmap_loading=original):
                assert flags.mmap_loading is original
            assert flags.mmap_loading is not original
        assert flags.mmap_loading is original


class TestListFlags:
    """list_flags() returns a plain dict of all current flag values."""

    def test_returns_dict(self):
        result = list_flags()
        assert isinstance(result, dict)

    def test_contains_all_flags(self):
        result = list_flags()
        assert "mmap_loading" in result
        assert "lazy_weight_loading" in result
        assert "suppress_dedup_warning" in result

    def test_values_match_singleton(self):
        result = list_flags()
        assert result["mmap_loading"] == flags.mmap_loading
        assert result["lazy_weight_loading"] == flags.lazy_weight_loading
        assert result["suppress_dedup_warning"] == flags.suppress_dedup_warning

    def test_returns_snapshot_not_live_view(self):
        """list_flags() returns a copy, not a live reference."""
        result = list_flags()
        original = flags.mmap_loading
        with override_flags(mmap_loading=not original):
            assert result["mmap_loading"] == original  # snapshot unchanged
