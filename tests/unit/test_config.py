"""Unit tests for config helpers."""

import logging

import pytest
import yaml

from plexe.config import (
    Config,
    DEFAULT_LLM,
    RoutingConfig,
    RoutingProviderConfig,
    get_routing_for_model,
    setup_logging,
)


def test_get_routing_for_model_mapping_and_default():
    """Mapped models use provider config; others use default."""
    config = RoutingConfig(
        default=RoutingProviderConfig(api_base="https://default", api_key="default-key", headers={"x": "1"}),
        providers={
            "p1": RoutingProviderConfig(api_base="https://p1", api_key="p1-key", headers={"y": "2"}),
        },
        models={"model-a": "p1"},
    )

    api_base, headers, api_key = get_routing_for_model(config, "model-a")
    assert api_base == "https://p1"
    assert headers == {"y": "2"}
    assert api_key == "p1-key"

    api_base, headers, api_key = get_routing_for_model(config, "model-b")
    assert api_base == "https://default"
    assert headers == {"x": "1"}
    assert api_key == "default-key"

    assert get_routing_for_model(None, "model-a") == (None, {}, None)


def test_get_llm_resolution(monkeypatch):
    """Per-role override beats global llm, which beats the built-in default."""
    for var in ("LLM", "PLEXE_LLM"):
        monkeypatch.delenv(var, raising=False)

    assert Config().get_llm("hypothesiser") == DEFAULT_LLM
    assert Config(llm="openai/gpt-6-luna").get_llm("hypothesiser") == "openai/gpt-6-luna"
    assert (
        Config(llm="openai/gpt-6-luna", hypothesiser_llm="anthropic/claude-test").get_llm("hypothesiser")
        == "anthropic/claude-test"
    )
    assert Config().get_llm("planner") == DEFAULT_LLM


def test_default_provider_is_opencode_go(monkeypatch):
    """Out of the box, agents use an OpenCode Go model routed to the Go gateway with the env key."""
    monkeypatch.setenv("OPENCODE_API_KEY", "oc_sk_test")
    monkeypatch.delenv("LLM", raising=False)
    monkeypatch.delenv("PLEXE_LLM", raising=False)

    config = Config()
    model = config.get_llm("planner")

    assert model == "openai/gpt-6-luna"
    assert get_routing_for_model(config.routing_config, model) == (
        "https://opencode.ai/inference/go/openai/v1",
        {},
        "oc_sk_test",
    )


def test_plexe_llm_env_fallback(monkeypatch):
    """PLEXE_LLM is honored when LLM is unset."""
    monkeypatch.delenv("LLM", raising=False)
    monkeypatch.setenv("PLEXE_LLM", "openai/glm-5.3")

    from plexe.config import get_config

    assert get_config().llm == "openai/glm-5.3"


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
