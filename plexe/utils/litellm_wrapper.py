"""
LiteLLM model wrapper with retry logic and optional post-call hook.

Provides automatic retry handling for transient errors and rate limits,
header injection for proxy authentication, and an optional callback hook
for observing LLM calls (usage tracking, logging, auditing, etc.).
"""

import logging
import time
from collections.abc import Callable
from typing import Any

from litellm import InternalServerError, APIConnectionError, RateLimitError, ServiceUnavailableError
from smolagents import LiteLLMModel
from tenacity import retry, stop_after_attempt, wait_exponential, wait_random, retry_if_exception_type

logger = logging.getLogger(__name__)


def _rate_limit_retry():
    """Retry strategy for rate limit errors: wait for window reset (~60s)."""
    return retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=1, min=60, max=300) + wait_random(0, 30),
        stop=stop_after_attempt(2),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            f"Rate limit hit, waiting before retry: {retry_state.outcome.exception()}"
        ),
    )


def _transient_error_retry():
    """Retry strategy for transient errors: exponential backoff ~5s, ~10s, ~20s, ~40s."""
    return retry(
        retry=retry_if_exception_type((InternalServerError, APIConnectionError, ServiceUnavailableError)),
        wait=wait_exponential(multiplier=2, min=5, max=60) + wait_random(0, 10),
        stop=stop_after_attempt(4),
        reraise=True,
        before_sleep=lambda retry_state: logger.info(f"Transient error, retrying: {retry_state.outcome.exception()}"),
    )


class PlexeLiteLLMModel(LiteLLMModel):
    """
    LiteLLM model wrapper with automatic retries and an optional post-call hook.

    Features:
    - Automatic retry with exponential backoff for transient errors and rate limits
    - Header injection for proxy/gateway authentication
    - Optional ``on_llm_call`` callback for observing LLM calls

    The ``on_llm_call`` callback receives the raw result object, enabling any
    downstream use case (token tracking, logging, auditing, content filtering)
    without coupling the wrapper to a specific implementation.

    Args:
        model_id: LiteLLM model identifier (e.g., ``"gpt-4"``, ``"claude-3-opus"``)
        extra_headers: Headers injected into every LLM call (e.g., proxy auth tokens)
        on_llm_call: Optional callback invoked after each successful LLM call.
            Signature: ``(model_id: str, result: Any, duration_ms: int) -> None``.
        **kwargs: Additional arguments passed to ``LiteLLMModel`` (e.g., ``api_base``)
    """

    def __init__(
        self,
        model_id: str,
        extra_headers: dict[str, str] | None = None,
        on_llm_call: Callable[[str, Any, int], None] | None = None,
        **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.extra_headers = extra_headers or {}
        self.on_llm_call = on_llm_call

    @_transient_error_retry()
    @_rate_limit_retry()
    def generate(self, *args, **kwargs):
        """Generate with automatic retries, header injection, and post-call hook."""
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers

        start_time = time.time()
        result = super().generate(*args, **kwargs)
        duration_ms = int((time.time() - start_time) * 1000)

        if self.on_llm_call:
            try:
                self.on_llm_call(self.model_id, result, duration_ms)
            except Exception as e:
                logger.warning(f"on_llm_call callback failed: {e}")

        return result

    @_transient_error_retry()
    @_rate_limit_retry()
    def chat(self, *args, **kwargs):
        """Chat with automatic retries, header injection, and post-call hook."""
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers

        start_time = time.time()
        result = super().chat(*args, **kwargs)
        duration_ms = int((time.time() - start_time) * 1000)

        if self.on_llm_call:
            try:
                self.on_llm_call(self.model_id, result, duration_ms)
            except Exception as e:
                logger.warning(f"on_llm_call callback failed: {e}")

        return result
