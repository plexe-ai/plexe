"""
Chain of thought capturing and logging for agent systems.

This package provides a framework-agnostic way to capture, format, and display
the chain of thought reasoning from different agent frameworks.
"""

from plexe.internal.common.utils.chain_of_thought.protocol import StepSummary, ToolCall
from plexe.internal.common.utils.chain_of_thought.adapters import extract_step_summary_from_smolagents
from plexe.internal.common.utils.chain_of_thought.callable import ChainOfThoughtCallable
from plexe.internal.common.utils.chain_of_thought.emitters import (
    ChainOfThoughtEmitter,
    ConsoleEmitter,
    LoggingEmitter,
    MultiEmitter,
)

__all__ = [
    "StepSummary",
    "ToolCall",
    "extract_step_summary_from_smolagents",
    "ChainOfThoughtCallable",
    "ChainOfThoughtEmitter",
    "ConsoleEmitter",
    "LoggingEmitter",
    "MultiEmitter",
]
