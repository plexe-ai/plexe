"""
This module defines the `Solution` class used to represent complete ML pipelines.

A Solution encapsulates all artifacts produced throughout the ML workflow for a single
approach, including the initial plan, feature transformations, model training code,
inference code, performance metrics, and deployment artifacts. Each Solution represents
one experimental path from data processing through model deployment.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import List, Dict

from plexe.internal.models.entities.artifact import Artifact
from plexe.internal.models.entities.metric import Metric


@dataclass(eq=False)
class Solution:
    """
    Represents a complete ML solution from planning through deployment.

    A Solution is a container for all artifacts related to a single ML approach,
    allowing different experiments to maintain their own schemas, transformations,
    and implementations. Solutions persist throughout the workflow and contain
    the final deployable model components.

    Attributes:
        id (str): A unique identifier for the solution.
        created_time (float): The UNIX timestamp when the solution was created.
        plan (str): The ML approach description and strategy for this solution.
        training_code (str): The model training implementation code.
        inference_code (str): The production inference/prediction code.
        input_schema (Dict[str, str]): Schema for the input data expected by the model.
        output_schema (Dict[str, str]): Schema for the output data produced by the model.
        performance (Metric): Validation set performance metrics.
        test_performance (Metric): Test set performance metrics.
        execution_time (float): Time taken to train the model.
        execution_stdout (list[str]): Training execution logs and output.
        exception_was_raised (bool): Whether training failed with an exception.
        exception (Exception): The exception raised during training, if any.
        model_artifacts (List[Path]): Paths to serialized model files and other artifacts.
        analysis (str): Summary analysis of the solution's effectiveness.
    """

    # General attributes
    id: str = field(default_factory=lambda: uuid.uuid4().hex, kw_only=True)
    created_time: float = field(default_factory=lambda: time.time(), kw_only=True)

    # Core solution attributes
    plan: str = field(default=None, hash=True, kw_only=True)
    training_code: str = field(default=None, hash=True, kw_only=True)
    inference_code: str = field(default=None, hash=True, kw_only=True)
    testing_code: str = field(default=None, hash=True, kw_only=True)
    input_schema: Dict[str, str] = field(default=None, kw_only=True)
    output_schema: Dict[str, str] = field(default=None, kw_only=True)
    schema_reasoning: str = field(default=None, kw_only=True)  # Explanation of schema design

    # Post-execution results: model performance, execution time, exceptions, etc.
    performance: Metric = field(default=None, kw_only=True)  # Validation performance
    test_performance: Metric = field(default=None, kw_only=True)  # Test set performance
    execution_time: float = field(default=None, kw_only=True)
    execution_stdout: list[str] = field(default_factory=list, kw_only=True)
    exception_was_raised: bool = field(default=False, kw_only=True)
    exception: Exception = field(default=None, kw_only=True)
    model_artifacts: List[Artifact] = field(default_factory=list, kw_only=True)

    # Evaluations and analyses
    analysis: str = field(default=None, kw_only=True)
    review: Dict[str, str] = field(default=None, kw_only=True)
    model_evaluation_report: Dict[str, any] = field(default_factory=dict, kw_only=True)
