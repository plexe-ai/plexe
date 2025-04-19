"""
This module defines the base class for LLM providers and includes
logging and retry mechanisms for querying the providers.
"""

import logging
import textwrap
from typing import Type, Optional

import litellm
from litellm import completion, supports_response_schema
from litellm.exceptions import RateLimitError, ServiceUnavailableError
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class ProviderConfig:
    """
    Configuration class for specifying different LLM providers for various agent roles.

    This allows for granular control of which providers/models are used for different
    parts of the multi-agent system.

    Attributes:
        default_provider: The default provider to use when specific ones aren't set
        orchestrator_provider: Provider for the orchestrator/manager agent
        research_provider: Provider for the ML Research Scientist agent
        engineer_provider: Provider for the ML Engineer agent
        ops_provider: Provider for the ML Ops Engineer agent
        tool_provider: Provider for tool operations
    """

    def __init__(
        self,
        default_provider: str = "openai/gpt-4o-mini",
        orchestrator_provider: Optional[str] = None,
        research_provider: Optional[str] = None,
        engineer_provider: Optional[str] = None,
        ops_provider: Optional[str] = None,
        tool_provider: Optional[str] = None,
    ):
        # Default provider is used when specific ones aren't set
        self.default_provider = default_provider

        # Agent-specific providers
        self.orchestrator_provider = orchestrator_provider or default_provider
        self.research_provider = research_provider or default_provider
        self.engineer_provider = engineer_provider or default_provider
        self.ops_provider = ops_provider or default_provider

        # Provider for tool operations
        self.tool_provider = tool_provider or default_provider

    def __repr__(self) -> str:
        return (
            f"ProviderConfig(default={self.default_provider}, "
            f"orchestrator={self.orchestrator_provider}, "
            f"research={self.research_provider}, "
            f"engineer={self.engineer_provider}, "
            f"ops={self.ops_provider}, "
            f"tool={self.tool_provider})"
        )


class Provider:
    """
    Base class for LiteLLM provider.
    """

    def __init__(self, model: str = None):
        default_model = "openai/gpt-4o-mini"
        self.model = model or default_model
        if "/" not in self.model:
            self.model = default_model
            logger.warning(f"Model name should be in the format 'provider/model', using default model: {default_model}")
        # Check if the model supports json mode
        if "response_format" not in litellm.get_supported_openai_params(model=self.model):
            raise ValueError(f"Model {self.model} does not support passing response_format")
        if not supports_response_schema(model=self.model):
            raise ValueError(f"Model {self.model} does not support response schema")

    def _make_completion_call(self, messages, response_format):
        """Helper method to make the actual API call with built-in retries for rate limits"""
        response = completion(model=self.model, messages=messages, response_format=response_format)

        if not response.choices[0].message.content:
            raise ValueError("Empty response from provider")

        return response.choices[0].message.content

    def query(
        self,
        system_message: str,
        user_message: str,
        response_format: Type[BaseModel] = None,
        retries: int = 3,
        backoff: bool = True,
    ) -> str:
        """
        Method to query the provider using litellm.completion.

        :param [str] system_message: The system message to send to the provider.
        :param [str] user_message: The user message to send to the provider.
        :param [Type[BaseModel]] response_format: A pydantic BaseModel class representing the response format.
        :param [int] retries: The number of times to retry the request. Defaults to 3.
        :param [bool] backoff: Whether to use exponential backoff when retrying. Defaults to True.
        :return [str]: The response from the provider.
        """
        self._log_request(system_message, user_message, self.__class__.__name__)

        messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]

        try:
            # Handle general errors with standard retries
            if backoff:

                @retry(stop=stop_after_attempt(retries), wait=wait_exponential(multiplier=2))
                def call_with_backoff_retry_all_errors():
                    @retry(
                        stop=stop_after_attempt(5),
                        wait=wait_exponential(multiplier=2, min=4),
                        retry=retry_if_exception_type((RateLimitError, ServiceUnavailableError)),
                    )
                    def call_with_backoff_retry_service_errors():
                        return self._make_completion_call(messages, response_format)

                    return call_with_backoff_retry_service_errors()

                r = call_with_backoff_retry_all_errors()
            else:
                r = self._make_completion_call(messages, response_format)

            self._log_response(r, self.__class__.__name__)
            return r
        except Exception as e:
            self._log_error(e)
            raise e

    @staticmethod
    def _log_request(system_message: str, user_message: str, model):
        """
        Logs the request to the provider.

        :param [str] system_message: The system message to send to the provider.
        :param [str] user_message: The user message to send to the provider.
        """
        logger.debug(
            (
                # String interpolation because Python <3.12 does not support backslashes inside f-strings curly braces
                f"Requesting chat completion from {model} with messages: "
                + textwrap.shorten(system_message.replace("\n", " "), 30)
                + ", "
                + textwrap.shorten(user_message.replace("\n", " "), 30)
            )
        )

    @staticmethod
    def _log_response(response, model):
        """
        Logs the response from the provider.

        :param [str] response: The response from the provider.
        """
        logger.debug(f"Received completion from {model}: {textwrap.shorten(response, 30)}")

    @staticmethod
    def _log_error(error):
        """
        Logs the error from the provider.

        :param [str] error: The error from the provider.
        """
        logger.error(f"Error querying provider: {error}")
