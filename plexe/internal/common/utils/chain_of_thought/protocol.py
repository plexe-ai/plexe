"""
Defines protocols and data classes for capturing agent reasoning steps.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class ToolCall:
    """Information about a tool called by an agent."""

    name: str
    args: Dict[str, Any]


@dataclass
class StepSummary:
    """
    Framework-agnostic representation of an agent's reasoning step.

    This class represents a single step in an agent's chain of thought,
    regardless of the underlying agent framework used.
    """

    step_number: Optional[int] = None
    step_type: str = "Unknown"
    agent_name: str = "Agent"
    model_output: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    observations: Optional[str] = None
    result: Any = None
    error: Optional[str] = None
    friendly_title: Optional[str] = None
    friendly_summary: Optional[str] = None


class StepExtractor(Protocol):
    """
    Protocol for extracting step information from agent frameworks.
    """

    def __call__(self, step: Any, agent: Any) -> StepSummary:
        """
        Extract step summary from framework-specific step object.

        Args:
            step: A step object from a specific agent framework
            agent: The agent that performed the step

        Returns:
            A framework-agnostic StepSummary object
        """
        ...
