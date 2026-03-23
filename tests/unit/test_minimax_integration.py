"""Integration tests for MiniMax provider support.

These tests verify end-to-end MiniMax configuration and routing.
They require a MINIMAX_API_KEY environment variable to run API calls.
"""

import os

import pytest

from plexe.config import Config, MINIMAX_API_BASE, get_routing_for_model


@pytest.fixture
def minimax_api_key():
    """Skip if MINIMAX_API_KEY is not set."""
    key = os.getenv("MINIMAX_API_KEY")
    if not key:
        pytest.skip("MINIMAX_API_KEY not set")
    return key


def test_minimax_config_yaml_roundtrip(tmp_path, monkeypatch):
    """Full config loading with MiniMax model IDs from YAML."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "hypothesiser_llm: minimax/MiniMax-M2.7\n"
        "planner_llm: minimax/MiniMax-M2.5-highspeed\n"
        "litellm_drop_params: true\n"
    )
    monkeypatch.setenv("CONFIG_FILE", str(config_path))
    monkeypatch.setenv("MINIMAX_API_KEY", "test-integration-key")

    config = Config()

    assert config.hypothesiser_llm == "minimax/MiniMax-M2.7"
    assert config.planner_llm == "minimax/MiniMax-M2.5-highspeed"
    assert config.litellm_drop_params is True

    # Verify routing resolves correctly
    api_base, headers = get_routing_for_model(config.routing_config, config.hypothesiser_llm)
    assert api_base == MINIMAX_API_BASE
    assert headers["Authorization"] == "Bearer test-integration-key"


def test_minimax_mixed_provider_config(tmp_path, monkeypatch):
    """Config with mixed providers (MiniMax + Anthropic) works correctly."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "hypothesiser_llm: minimax/MiniMax-M2.7\n"
        "planner_llm: anthropic/claude-sonnet-4-5-20250929\n"
        "model_definer_llm: minimax/MiniMax-M2.5-highspeed\n"
    )
    monkeypatch.setenv("CONFIG_FILE", str(config_path))
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")

    config = Config()

    # MiniMax models auto-route to MiniMax API
    api_base, headers = get_routing_for_model(config.routing_config, config.hypothesiser_llm)
    assert api_base == MINIMAX_API_BASE

    # Anthropic models use default (None) routing
    api_base, headers = get_routing_for_model(config.routing_config, config.planner_llm)
    assert api_base is None
    assert headers == {}


def test_minimax_api_completion(minimax_api_key):
    """Verify MiniMax API responds to a basic completion request via LiteLLM."""
    import litellm

    response = litellm.completion(
        model="openai/MiniMax-M2.5-highspeed",
        messages=[{"role": "user", "content": "Say hello in one word."}],
        api_base=MINIMAX_API_BASE,
        api_key=minimax_api_key,
        temperature=0.2,
        max_tokens=256,
    )

    message = response.choices[0].message
    # MiniMax models may include reasoning_content alongside content
    has_content = message.content and len(message.content.strip()) > 0
    has_reasoning = getattr(message, "reasoning_content", None) and len(message.reasoning_content.strip()) > 0
    assert has_content or has_reasoning, f"Expected non-empty response, got content={message.content!r}"
