"""Build LiteLLM model instances with per-model routing applied."""

from plexe.config import Config, get_routing_for_model
from plexe.utils.litellm_wrapper import PlexeLiteLLMModel


def build_llm_model(config: Config, llm_model: str, temperature: float) -> PlexeLiteLLMModel:
    """
    Build a LiteLLM model with routing configuration applied.

    Args:
        config: Workflow configuration (provides routing)
        llm_model: LiteLLM model ID, e.g. ``"openai/gpt-6-luna"``
        temperature: Sampling temperature for the agent

    Returns:
        Configured ``PlexeLiteLLMModel``
    """
    api_base, headers, api_key = get_routing_for_model(config.routing_config, llm_model)
    return PlexeLiteLLMModel(
        model_id=llm_model,
        temperature=temperature,
        api_base=api_base,
        api_key=api_key,
        extra_headers=headers,
    )
