"""Unit tests for PlexeLiteLLMModel MiniMax support."""

from unittest.mock import patch, MagicMock

import pytest

from plexe.utils.litellm_wrapper import PlexeLiteLLMModel


@pytest.fixture(autouse=True)
def _patch_litellm_model():
    """Patch LiteLLMModel.__init__ so tests don't need real LiteLLM credentials."""
    with patch("plexe.utils.litellm_wrapper.LiteLLMModel.__init__", return_value=None):
        yield


class TestMiniMaxModelIdRewriting:
    """Tests for minimax/ prefix → openai/ rewriting in PlexeLiteLLMModel."""

    def test_minimax_prefix_rewritten_to_openai(self):
        """minimax/MiniMax-M2.7 should become openai/MiniMax-M2.7."""
        model = PlexeLiteLLMModel(model_id="minimax/MiniMax-M2.7", temperature=0.2)
        # The super().__init__ was patched, so check the call args
        from plexe.utils.litellm_wrapper import LiteLLMModel

        LiteLLMModel.__init__.assert_called_once_with(model_id="openai/MiniMax-M2.7", temperature=0.2)

    def test_minimax_highspeed_rewritten(self):
        """minimax/MiniMax-M2.5-highspeed should become openai/MiniMax-M2.5-highspeed."""
        model = PlexeLiteLLMModel(model_id="minimax/MiniMax-M2.5-highspeed", temperature=0.5)
        from plexe.utils.litellm_wrapper import LiteLLMModel

        LiteLLMModel.__init__.assert_called_once_with(model_id="openai/MiniMax-M2.5-highspeed", temperature=0.5)

    def test_non_minimax_model_unchanged(self):
        """openai/gpt-4 should not be rewritten."""
        model = PlexeLiteLLMModel(model_id="openai/gpt-4", temperature=0.7)
        from plexe.utils.litellm_wrapper import LiteLLMModel

        LiteLLMModel.__init__.assert_called_once_with(model_id="openai/gpt-4", temperature=0.7)

    def test_anthropic_model_unchanged(self):
        """anthropic/ models should not be rewritten."""
        model = PlexeLiteLLMModel(model_id="anthropic/claude-sonnet-4-5-20250929", temperature=0.2)
        from plexe.utils.litellm_wrapper import LiteLLMModel

        LiteLLMModel.__init__.assert_called_once_with(
            model_id="anthropic/claude-sonnet-4-5-20250929", temperature=0.2
        )


class TestMiniMaxTemperatureClamping:
    """Tests for MiniMax temperature clamping in PlexeLiteLLMModel."""

    def test_minimax_temperature_clamped_high(self):
        """Temperature > 1.0 should be clamped to 1.0 for MiniMax models."""
        model = PlexeLiteLLMModel(model_id="minimax/MiniMax-M2.7", temperature=1.5)
        from plexe.utils.litellm_wrapper import LiteLLMModel

        LiteLLMModel.__init__.assert_called_once_with(model_id="openai/MiniMax-M2.7", temperature=1.0)

    def test_minimax_temperature_clamped_low(self):
        """Temperature < 0.0 should be clamped to 0.0 for MiniMax models."""
        model = PlexeLiteLLMModel(model_id="minimax/MiniMax-M2.7", temperature=-0.1)
        from plexe.utils.litellm_wrapper import LiteLLMModel

        LiteLLMModel.__init__.assert_called_once_with(model_id="openai/MiniMax-M2.7", temperature=0.0)

    def test_minimax_temperature_within_range_unchanged(self):
        """Temperature within [0, 1.0] should not be modified for MiniMax models."""
        model = PlexeLiteLLMModel(model_id="minimax/MiniMax-M2.7", temperature=0.7)
        from plexe.utils.litellm_wrapper import LiteLLMModel

        LiteLLMModel.__init__.assert_called_once_with(model_id="openai/MiniMax-M2.7", temperature=0.7)

    def test_non_minimax_temperature_not_clamped(self):
        """Temperature > 1.0 should NOT be clamped for non-MiniMax models."""
        model = PlexeLiteLLMModel(model_id="openai/gpt-4", temperature=1.5)
        from plexe.utils.litellm_wrapper import LiteLLMModel

        LiteLLMModel.__init__.assert_called_once_with(model_id="openai/gpt-4", temperature=1.5)

    def test_minimax_no_temperature_kwarg(self):
        """When no temperature is provided, no clamping should occur."""
        model = PlexeLiteLLMModel(model_id="minimax/MiniMax-M2.7")
        from plexe.utils.litellm_wrapper import LiteLLMModel

        LiteLLMModel.__init__.assert_called_once_with(model_id="openai/MiniMax-M2.7")

    def test_minimax_api_base_passthrough(self):
        """api_base kwarg should be passed through to LiteLLMModel."""
        model = PlexeLiteLLMModel(
            model_id="minimax/MiniMax-M2.7",
            temperature=0.2,
            api_base="https://api.minimax.io/v1",
        )
        from plexe.utils.litellm_wrapper import LiteLLMModel

        LiteLLMModel.__init__.assert_called_once_with(
            model_id="openai/MiniMax-M2.7",
            temperature=0.2,
            api_base="https://api.minimax.io/v1",
        )
