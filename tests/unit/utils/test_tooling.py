"""Unit tests for tooling utilities."""

import pytest

from plexe.utils.tooling import AgentInvocationError, agentinspectable


@agentinspectable
def greet(name: str, times: int = 1):
    """greet(name: str, times: int = 1) -> str"""
    return " ".join([f"hi {name}"] * times)


def test_agentinspectable_valid_call():
    """Decorator should allow valid calls."""
    assert greet("marc", 2) == "hi marc hi marc"


def test_agentinspectable_invalid_call_raises():
    """Invalid calls should raise AgentInvocationError with usage."""
    with pytest.raises(AgentInvocationError) as exc:
        greet()

    message = str(exc.value)
    assert "Usage:" in message
    assert "greet" in message
