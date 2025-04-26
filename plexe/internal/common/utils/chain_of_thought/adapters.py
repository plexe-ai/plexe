"""
This module provides adapters for extracting step information from different agent frameworks.

The functions in this module are designed to take as input the step outputs provided by a particular
agent framework, and convert them into a framework-agnostic representation that is used throughout Plexe.
This module should be extended as new agent frameworks are added to Plexe, or modified if the agent
frameworks change their output formats.
"""

from typing import Any

from .protocol import StepSummary, ToolCall


def extract_step_summary_from_smolagents(step: Any, agent: Any) -> StepSummary:
    """
    Extract step summary from a SmoLAgents step object.

    Args:
        step: A SmoLAgents step object
        agent: The agent that performed the step

    Returns:
        A framework-agnostic StepSummary object
    """
    # Get agent name
    agent_name = getattr(agent, "name", agent.__class__.__name__)

    # Extract common properties
    step_number = getattr(step, "step_number", None)
    step_type = step.__class__.__name__
    error = str(getattr(step, "error", "")) or None

    # Extract model output from various step types
    model_output = None
    if hasattr(step, "model_output_message") and step.model_output_message:
        model_output = getattr(step.model_output_message, "content", None)

    # Extract tool calls
    tool_calls = []
    if hasattr(step, "tool_calls") and step.tool_calls:
        tool_calls = [ToolCall(name=call.name, args=call.arguments) for call in step.tool_calls]

    # Extract observations and results based on step type
    observations = getattr(step, "observations", None)
    result = getattr(step, "action_output", None)

    # Handle specific step types
    if hasattr(step, "code_block") and getattr(step, "code_block", None):
        # Extract thought and code from CodeActionStep
        code_block = getattr(step, "code_block", None)
        if code_block:
            # Don't show full code in chain of thought, just indicate it's there
            observations = f"[Code block executed: {len(code_block)} characters]"

    # Create and return the step summary
    return StepSummary(
        step_number=step_number,
        step_type=step_type,
        agent_name=agent_name,
        model_output=model_output,
        tool_calls=tool_calls,
        observations=observations,
        result=result,
        error=error,
    )
