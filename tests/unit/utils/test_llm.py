"""Unit tests for the LLM model factory."""

from unittest.mock import patch

from plexe.config import Config, RoutingConfig, RoutingProviderConfig
from plexe.utils.llm import build_llm_model


def test_build_llm_model_applies_routing():
    """Factory forwards model id, temperature, and routing (incl. api_key)."""
    config = Config(
        routing_config=RoutingConfig(
            default=RoutingProviderConfig(
                api_base="https://opencode.ai/inference/go/openai/v1",
                api_key="secret",
                headers={"x": "1"},
            )
        )
    )

    with patch("plexe.utils.llm.PlexeLiteLLMModel") as model_cls:
        build_llm_model(config, "openai/gpt-6-luna", 0.3)

    model_cls.assert_called_once_with(
        model_id="openai/gpt-6-luna",
        temperature=0.3,
        api_base="https://opencode.ai/inference/go/openai/v1",
        api_key="secret",
        extra_headers={"x": "1"},
    )
