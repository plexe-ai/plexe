"""
This module defines Callables for capturing and formatting agent chain of thought.

The classes in this module are designed to be used as "plug-ins" to agent frameworks, enabling the production
of a user-friendly output of the agent's reasoning process. The output can be used for debugging, logging, or
user feedback during agent execution.
"""

import json
import logging
from typing import Any, List, Optional, Tuple

from pydantic import BaseModel, Field

from ...provider import Provider
from plexe.config import prompt_templates
from .protocol import StepExtractor, StepSummary
from .adapters import extract_step_summary_from_smolagents
from .emitters import ChainOfThoughtEmitter, ConsoleEmitter

logger = logging.getLogger(__name__)


class ChainOfThoughtCallable:
    """
    Callable that captures and formats agent chain of thought.

    This callable can be attached to agent frameworks to capture
    each step of the agent's reasoning process and format it for
    user-friendly output.
    """

    def __init__(
        self,
        emitter: Optional[ChainOfThoughtEmitter] = None,
        extractor: StepExtractor = extract_step_summary_from_smolagents,
    ):
        """
        Initialize the chain of thought callable.

        Args:
            emitter: The emitter to use for outputting chain of thought
            extractor: Function that extracts step information from the agent framework
        """
        self.emitter = emitter or ConsoleEmitter()
        self.extractor = extractor
        self.steps: List[StepSummary] = []

    def __call__(self, step: Any, agent: Any = None) -> None:
        """
        Process a step from an agent.

        Args:
            step: The step object from the agent framework
            agent: The agent that performed the step
        """
        try:
            # Extract step summary
            summary = self.extractor(step, agent)

            # Generate friendly title and summary if not already present
            if summary.friendly_title is None or summary.friendly_summary is None:
                friendly_title, friendly_summary = _generate_friendly_summary(summary)
                summary.friendly_title = friendly_title
                summary.friendly_summary = friendly_summary

            # Store the step for later retrieval
            self.steps.append(summary)

            # Emit the step information
            self._emit_step(summary)

        except Exception as e:
            # Log full stack trace at debug level
            import traceback

            logger.debug(f"Error processing agent step: {str(e)}\n{traceback.format_exc()}")

            # Log a shorter message at warning level
            logger.warning(f"Error processing agent step: {str(e)[:50]}")

    def _emit_step(self, summary: StepSummary) -> None:
        """
        Emit a step to the configured emitter.

        Args:
            summary: The step summary to emit
        """
        # If we have friendly title and summary, emit those in a consolidated message
        if summary.friendly_title and summary.friendly_summary:
            # Emit friendly step header with title and summary together
            # This ensures they appear as a single node in the tree
            self.emitter.emit_thought(summary.agent_name, f"💡 {summary.friendly_title}\n💭 {summary.friendly_summary}")
            return

        # Fall back to verbose output if friendly version not available

        # Emit step header
        step_header = f"🧠 {summary.step_type}"
        if summary.step_number is not None:
            step_header += f" #{summary.step_number}"

        self.emitter.emit_thought(summary.agent_name, step_header)

        # Emit model output separately if available
        if summary.model_output:
            thought_text = summary.model_output[:500]
            if len(summary.model_output) > 500:
                thought_text += "..."
            self.emitter.emit_thought(summary.agent_name, f"💭 Thought: {thought_text}")

        # Emit tool calls one by one for better visualization
        for call in summary.tool_calls:
            self.emitter.emit_thought(summary.agent_name, f"🔧 Tool: {call.name}({call.args})")

        # Emit observations
        if summary.observations:
            observation_text = summary.observations[:500]
            if len(summary.observations) > 500:
                observation_text += "..."
            self.emitter.emit_thought(summary.agent_name, f"📡 Observed: {observation_text}")

        # Emit result
        if summary.result:
            result_text = str(summary.result)[:500]
            if len(str(summary.result)) > 500:
                result_text += "..."
            self.emitter.emit_thought(summary.agent_name, f"📦 Result: {result_text}")

        # Emit error if any
        if summary.error:
            self.emitter.emit_thought(summary.agent_name, f"❌ Error: {summary.error}")

    def get_full_chain_of_thought(self) -> List[StepSummary]:
        """
        Get the full chain of thought captured so far.

        Returns:
            The list of step summaries
        """
        return self.steps

    def clear(self) -> None:
        """Clear all captured steps."""
        self.steps = []


def _generate_friendly_summary(summary: StepSummary) -> Tuple[str, str]:
    """
    Generate a user-friendly title and summary for a step using LLM.

    Args:
        summary: The step summary to generate a friendly summary for

    Returns:
        A tuple of (friendly_title, friendly_summary)
    """

    class FriendlySummaryResponse(BaseModel):
        """Response format for generating friendly step summaries."""

        title: str = Field(description="A short, friendly title (3-7 words) that captures the essence of what happened")
        summary: str = Field(description="A concise summary (1-2 sentences) that explains the step in plain language")

    # Create a context string that summarizes the step for the LLM
    context_parts = [f"Step Type: {summary.step_type}"]

    if summary.model_output:
        context_parts.append(f"Thought: {summary.model_output}")

    if summary.tool_calls:
        for call in summary.tool_calls:
            context_parts.append(f"Tool: {call.name}({json.dumps(call.args)})")

    if summary.observations:
        context_parts.append(f"Observation: {summary.observations}")

    if summary.result:
        context_parts.append(f"Result: {str(summary.result)}")

    if summary.error:
        context_parts.append(f"Error: {summary.error}")

    context = "\n".join(context_parts)

    # Get the prompt template
    system_message = prompt_templates.cot_system()
    user_message = prompt_templates.cot_summarize(context)

    try:
        # Use the Provider to get a structured response
        provider = Provider()
        response = provider.query(
            system_message=system_message, user_message=user_message, response_format=FriendlySummaryResponse
        )

        # Parse the response to get JSON
        response_data = json.loads(response)
        return response_data["title"], response_data["summary"]
    except Exception as e:
        # Log full stack trace at debug level
        import traceback

        logger.debug(f"Error generating friendly summary: {str(e)}\n{traceback.format_exc()}")

        # Log shorter message at warning level
        logger.warning(f"Error generating friendly summary: {str(e)[:50]}")

        return f"{summary.step_type}", f"Step {summary.step_number or 0} of type {summary.step_type}"
