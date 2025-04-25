"""
This module defines Callables for capturing and formatting agent chain of thought.

The classes in this module are designed to be used as "plug-ins" to agent frameworks, enabling the production
of a user-friendly output of the agent's reasoning process. The output can be used for debugging, logging, or
user feedback during agent execution.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

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
            
            # Store the step for later retrieval
            self.steps.append(summary)
            
            # Emit the step information
            self._emit_step(summary)
            
        except Exception as e:
            logger.warning(f"Error processing agent step: {str(e)}")
    
    def _emit_step(self, summary: StepSummary) -> None:
        """
        Emit a step to the configured emitter.
        
        Args:
            summary: The step summary to emit
        """
        # Emit step header
        self.emitter.emit_thought(
            summary.agent_name, 
            f"🧠 {summary.step_type} #{summary.step_number}"
        )
        
        # Emit model output if available
        if summary.model_output:
            self.emitter.emit_thought(
                summary.agent_name,
                f"💭 Thought: {summary.model_output[:200]}{'...' if len(summary.model_output) > 200 else ''}"
            )
        
        # Emit tool calls
        for call in summary.tool_calls:
            self.emitter.emit_thought(
                summary.agent_name,
                f"🔧 Tool: {call.name}({call.args})"
            )
        
        # Emit observations
        if summary.observations:
            self.emitter.emit_thought(
                summary.agent_name,
                f"📡 Observed: {summary.observations[:200]}{'...' if len(summary.observations) > 200 else ''}"
            )
        
        # Emit result
        if summary.result:
            self.emitter.emit_thought(
                summary.agent_name,
                f"📦 Result: {str(summary.result)[:200]}{'...' if len(str(summary.result)) > 200 else ''}"
            )
        
        # Emit error if any
        if summary.error:
            self.emitter.emit_thought(
                summary.agent_name,
                f"❌ Error: {summary.error}"
            )
    
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