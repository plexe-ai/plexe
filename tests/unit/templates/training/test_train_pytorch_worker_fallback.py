"""Unit tests for PyTorch DataLoader worker fallback behavior."""

import pytest

pytest.importorskip("torch")

from plexe.templates.training import train_pytorch


def test_resolve_num_workers_zero_is_unchanged() -> None:
    """Requested zero workers should remain zero."""
    assert train_pytorch._resolve_num_workers(0) == 0


def test_resolve_num_workers_falls_back_on_darwin_spawn(monkeypatch) -> None:
    """On macOS spawn, requested workers should fall back to zero."""
    monkeypatch.setattr(train_pytorch.sys, "platform", "darwin")
    monkeypatch.setattr(train_pytorch.mp, "get_start_method", lambda allow_none=True: "spawn")
    assert train_pytorch._resolve_num_workers(4) == 0


def test_resolve_num_workers_uses_context_when_start_method_is_none(monkeypatch) -> None:
    """When get_start_method returns None, context start method should be used."""

    class _Context:
        @staticmethod
        def get_start_method() -> str:
            return "spawn"

    monkeypatch.setattr(train_pytorch.sys, "platform", "darwin")
    monkeypatch.setattr(train_pytorch.mp, "get_start_method", lambda allow_none=True: None)
    monkeypatch.setattr(train_pytorch.mp, "get_context", lambda: _Context())
    assert train_pytorch._resolve_num_workers(2) == 0


def test_resolve_num_workers_kept_on_non_darwin_spawn(monkeypatch) -> None:
    """Spawn on non-macOS should keep the requested worker count."""
    monkeypatch.setattr(train_pytorch.sys, "platform", "linux")
    monkeypatch.setattr(train_pytorch.mp, "get_start_method", lambda allow_none=True: "spawn")
    assert train_pytorch._resolve_num_workers(3) == 3
