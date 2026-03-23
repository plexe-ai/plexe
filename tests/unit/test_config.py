"""Unit tests for config helpers."""

import logging

import pytest
import yaml

from plexe.config import (
    Config,
    MINIMAX_API_BASE,
    MINIMAX_MODELS,
    RoutingConfig,
    RoutingProviderConfig,
    _is_minimax_model,
    get_routing_for_model,
    setup_logging,
)


def test_get_routing_for_model_mapping_and_default():
    """Mapped models use provider config; others use default."""
    config = RoutingConfig(
        default=RoutingProviderConfig(api_base="https://default", headers={"x": "1"}),
        providers={
            "p1": RoutingProviderConfig(api_base="https://p1", headers={"y": "2"}),
        },
        models={"model-a": "p1"},
    )

    api_base, headers = get_routing_for_model(config, "model-a")
    assert api_base == "https://p1"
    assert headers == {"y": "2"}

    api_base, headers = get_routing_for_model(config, "model-b")
    assert api_base == "https://default"
    assert headers == {"x": "1"}


def test_temperature_fields_from_env(monkeypatch):
    monkeypatch.setenv("DEFAULT_TEMPERATURE", "0.15")
    monkeypatch.setenv("HYPOTHESISER_TEMPERATURE", "0.65")
    monkeypatch.setenv("PLANNER_TEMPERATURE", "0.66")
    monkeypatch.setenv("INSIGHT_EXTRACTOR_TEMPERATURE", "0.55")

    config = Config()

    assert config.default_temperature == 0.15
    assert config.hypothesiser_temperature == 0.65
    assert config.planner_temperature == 0.66
    assert config.insight_extractor_temperature == 0.55


def test_temperature_fields_from_yaml(tmp_path, monkeypatch):
    monkeypatch.delenv("DEFAULT_TEMPERATURE", raising=False)
    monkeypatch.delenv("HYPOTHESISER_TEMPERATURE", raising=False)
    monkeypatch.delenv("PLANNER_TEMPERATURE", raising=False)
    monkeypatch.delenv("INSIGHT_EXTRACTOR_TEMPERATURE", raising=False)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_temperature": 0.12,
                "hypothesiser_temperature": 0.62,
                "planner_temperature": 0.63,
                "insight_extractor_temperature": 0.52,
            }
        )
    )

    monkeypatch.setenv("CONFIG_FILE", str(config_path))

    config = Config()

    assert config.default_temperature == 0.12
    assert config.hypothesiser_temperature == 0.62
    assert config.planner_temperature == 0.63
    assert config.insight_extractor_temperature == 0.52


def test_get_temperature_resolves_override_and_default():
    config = Config(default_temperature=0.2, hypothesiser_temperature=0.7)

    assert config.get_temperature("hypothesiser") == pytest.approx(0.7)
    assert config.get_temperature("layout_detector") == pytest.approx(0.2)


def test_setup_logging_disables_propagation():
    """Plexe logger should not propagate to root to avoid duplicate log lines."""
    logger = setup_logging(Config(log_level="INFO"))

    assert logger.name == "plexe"
    assert logger.propagate is False
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)


# ============================================
# MiniMax Provider Tests
# ============================================


def test_is_minimax_model():
    """minimax/ prefix is correctly detected."""
    assert _is_minimax_model("minimax/MiniMax-M2.7") is True
    assert _is_minimax_model("minimax/MiniMax-M2.5-highspeed") is True
    assert _is_minimax_model("openai/gpt-4") is False
    assert _is_minimax_model("anthropic/claude-sonnet-4-5-20250929") is False


def test_minimax_models_constant():
    """MINIMAX_MODELS should contain expected model entries."""
    assert "MiniMax-M2.7" in MINIMAX_MODELS
    assert "MiniMax-M2.7-highspeed" in MINIMAX_MODELS
    assert "MiniMax-M2.5" in MINIMAX_MODELS
    assert "MiniMax-M2.5-highspeed" in MINIMAX_MODELS
    assert MINIMAX_MODELS["MiniMax-M2.7"]["context_window"] == 1_000_000


def test_minimax_auto_routing_no_config(monkeypatch):
    """minimax/ models auto-route even without routing_config."""
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key-123")

    api_base, headers = get_routing_for_model(None, "minimax/MiniMax-M2.7")

    assert api_base == MINIMAX_API_BASE
    assert headers == {"Authorization": "Bearer test-key-123"}


def test_minimax_auto_routing_with_config(monkeypatch):
    """minimax/ models auto-route when not explicitly mapped in config."""
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key-456")
    config = RoutingConfig(
        default=RoutingProviderConfig(api_base="https://default", headers={"x": "1"}),
    )

    api_base, headers = get_routing_for_model(config, "minimax/MiniMax-M2.7")

    assert api_base == MINIMAX_API_BASE
    assert headers == {"Authorization": "Bearer test-key-456"}


def test_minimax_explicit_mapping_overrides_auto_routing(monkeypatch):
    """Explicit routing_config mapping takes priority over minimax auto-routing."""
    monkeypatch.setenv("MINIMAX_API_KEY", "should-not-be-used")
    config = RoutingConfig(
        providers={
            "my-proxy": RoutingProviderConfig(api_base="https://proxy.example.com/v1", headers={"auth": "proxy-key"}),
        },
        models={"minimax/MiniMax-M2.7": "my-proxy"},
    )

    api_base, headers = get_routing_for_model(config, "minimax/MiniMax-M2.7")

    assert api_base == "https://proxy.example.com/v1"
    assert headers == {"auth": "proxy-key"}


def test_minimax_auto_routing_without_api_key(monkeypatch):
    """minimax/ models auto-route without API key (headers empty)."""
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)

    api_base, headers = get_routing_for_model(None, "minimax/MiniMax-M2.5-highspeed")

    assert api_base == MINIMAX_API_BASE
    assert headers == {}


def test_minimax_llm_from_yaml(tmp_path, monkeypatch):
    """MiniMax model IDs can be loaded from YAML config."""
    monkeypatch.delenv("HYPOTHESISER_LLM", raising=False)
    monkeypatch.delenv("PLANNER_LLM", raising=False)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "hypothesiser_llm": "minimax/MiniMax-M2.7",
                "planner_llm": "minimax/MiniMax-M2.5-highspeed",
            }
        )
    )
    monkeypatch.setenv("CONFIG_FILE", str(config_path))

    config = Config()

    assert config.hypothesiser_llm == "minimax/MiniMax-M2.7"
    assert config.planner_llm == "minimax/MiniMax-M2.5-highspeed"


def test_minimax_llm_from_env(monkeypatch):
    """MiniMax model IDs can be set via environment variables."""
    monkeypatch.setenv("HYPOTHESISER_LLM", "minimax/MiniMax-M2.7")

    config = Config()

    assert config.hypothesiser_llm == "minimax/MiniMax-M2.7"


def test_non_minimax_routing_unchanged():
    """Non-minimax models still use default routing behavior."""
    config = RoutingConfig(
        default=RoutingProviderConfig(api_base="https://default", headers={"x": "1"}),
    )

    api_base, headers = get_routing_for_model(config, "openai/gpt-4")

    assert api_base == "https://default"
    assert headers == {"x": "1"}


def test_non_minimax_no_config_returns_none():
    """Non-minimax models without config return (None, {})."""
    api_base, headers = get_routing_for_model(None, "openai/gpt-4")

    assert api_base is None
    assert headers == {}
