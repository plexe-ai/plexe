"""Unit tests for config helpers."""

import pytest
import yaml

from plexe.config import Config, RoutingConfig, RoutingProviderConfig, get_routing_for_model


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
