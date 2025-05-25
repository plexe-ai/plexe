"""
Callbacks for model building process in Plexe.

This module defines callback interfaces that let users hook into various stages
of the model building process, allowing for custom logging, tracking, visualization,
or other operations to be performed at key points.
"""

import logging
from abc import ABC
from dataclasses import dataclass
from typing import Optional, Type, Dict, Any

from pydantic import BaseModel

from plexe.internal.common.datasets.interface import TabularConvertible
from plexe.internal.models.entities.node import Node

logger = logging.getLogger(__name__)


@dataclass
class BuildStateInfo:
    """
    Consolidated information about model build state at any point in the process.

    This class combines all information available during different stages of the model building
    process (start, end, iteration start, iteration end) into a single structure.
    """

    # Common identification fields
    intent: str
    """The natural language description of the model's intent."""

    provider: str
    """The provider (LLM) used for generating the model."""

    # Schema fields
    input_schema: Optional[Type[BaseModel]] = None
    """The input schema for the model."""

    output_schema: Optional[Type[BaseModel]] = None
    """The output schema for the model."""

    run_timeout: Optional[int] = None
    """Maximum time in seconds for each individual training run."""

    max_iterations: Optional[int] = None
    """Maximum number of iterations for the model building process."""

    timeout: Optional[int] = None
    """Maximum total time in seconds for the entire model building process."""

    # Iteration fields
    iteration: int = 0
    """Current iteration number (0-indexed)."""

    # Dataset fields
    datasets: Optional[Dict[str, TabularConvertible]] = None

    # Current node being evaluated
    node: Optional[Node] = None
    """The solution node being evaluated in the current iteration."""

    # Model information fields (replacing direct model reference)
    model_identifier: Optional[str] = None
    """Model unique identifier."""

    model_state: Optional[str] = None
    """Current model state (BUILDING/READY/ERROR)."""

    # Final model artifacts (only available at build end)
    final_metric: Optional[Any] = None
    """Final performance metric."""

    final_artifacts: Optional[list] = None
    """Model artifacts list."""

    trainer_source: Optional[str] = None
    """Training source code."""

    predictor_source: Optional[str] = None
    """Predictor source code."""


class Callback(ABC):
    """
    Abstract base class for callbacks during model building.

    Callbacks allow running custom code at various stages of the model building process.
    Subclass this and implement the methods you need for your specific use case.
    """

    def on_build_start(self, info: BuildStateInfo) -> None:
        """
        Called when the model building process starts.
        """
        pass

    def on_build_end(self, info: BuildStateInfo) -> None:
        """
        Called when the model building process ends.
        """
        pass

    def on_iteration_start(self, info: BuildStateInfo) -> None:
        """
        Called at the start of each model building iteration.
        """
        pass

    def on_iteration_end(self, info: BuildStateInfo) -> None:
        """
        Called at the end of each model building iteration.
        """
        pass


# Import at the end to avoid circular dependencies
from plexe.internal.models.callbacks.mlflow import MLFlowCallback
from plexe.internal.models.callbacks.chain_of_thought import ChainOfThoughtModelCallback
from plexe.internal.models.callbacks.checkpoint import ModelCheckpointCallback

__all__ = [
    "Callback",
    "BuildStateInfo",
    "MLFlowCallback",
    "ChainOfThoughtModelCallback",
    "ModelCheckpointCallback",
]
