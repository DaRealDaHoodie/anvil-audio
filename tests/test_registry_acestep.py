"""Tests for ACE-Step registry entry with optional project_root."""
import os
import pytest
from anvil_audio.core.registry import RegistryEntry


def test_acestep_entry_registers_without_project_root(monkeypatch):
    """RegistryEntry for acestep registers when project_root is None and no env var."""
    monkeypatch.delenv("ACESTEP_PROJECT_ROOT", raising=False)
    entry = RegistryEntry(
        name="test-acestep",
        pipeline_type="acestep",
        acestep_project_root=None,
    )
    assert entry.name == "test-acestep"


def test_acestep_entry_uses_env_var_fallback(monkeypatch, tmp_path):
    """RegistryEntry picks up ACESTEP_PROJECT_ROOT env var when project_root is None."""
    monkeypatch.setenv("ACESTEP_PROJECT_ROOT", str(tmp_path))
    entry = RegistryEntry(
        name="test-acestep",
        pipeline_type="acestep",
        acestep_project_root=None,
    )
    assert entry.acestep_project_root == str(tmp_path)


def test_acestep_entry_explicit_root_takes_priority(monkeypatch, tmp_path):
    """Explicit project_root overrides env var."""
    monkeypatch.setenv("ACESTEP_PROJECT_ROOT", "/should/be/ignored")
    entry = RegistryEntry(
        name="test-acestep",
        pipeline_type="acestep",
        acestep_project_root=str(tmp_path),
    )
    assert entry.acestep_project_root == str(tmp_path)
