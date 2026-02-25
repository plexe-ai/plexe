"""Unit tests for config helpers."""

from plexe.config import RoutingConfig, RoutingProviderConfig, get_routing_for_model


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
