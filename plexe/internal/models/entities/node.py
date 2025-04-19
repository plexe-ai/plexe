"""
This module defines the `Node` and `Edge` classes used to represent directed graphs.

Nodes represent individual solutions or states in a problem-solving graph, while edges
represent directed relationships between nodes. The `Node` class tracks various
attributes related to the solution, such as generated code, performance metrics, and
execution details.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import List
from pathlib import Path

from plexe.internal.models.entities.metric import Metric


@dataclass(eq=False)
class Edge:
    """
    Represents a directed edge between two nodes in a graph.

    Attributes:
        source (Node): The source node of the edge.
        target (Node): The target node of the edge.
    """

    source: "Node" = field(kw_only=True)
    target: "Node" = field(kw_only=True)


@dataclass(eq=False)
class Node:
    """
    Represents a node in a directed graph, used to model solutions or states in a workflow.

    Attributes:
        id (str): A unique identifier for the node.
        created_time (float): The UNIX timestamp when the node was created.
        edges_in (List[Edge]): A list of edges pointing to this node.
        edges_out (List[Edge]): A list of edges originating from this node.
        solution_plan (str): The plan or description of the solution represented by this node.
        training_code (str): The code used for training models in this node.
        inference_code (str): The code used for inference in this node.
        training_tests (str): The code used for testing the training solution in this node.
        inference_tests (str): The code used for testing the inference solution in this node.
        estimated_value (float): The estimated value or utility of this node.
        estimated_cost (float): The estimated cost associated with this node.
        performance (List[Metric]): A list of metrics evaluating the solution's performance.
        execution_time (float): The time taken to execute the solution code.
        execution_stdout (list[str]): The standard output from the solution's execution.
        exception_was_raised (bool): Indicates whether an exception occurred during execution.
        exception (Exception): The exception raised during execution, if any.
        model_artifacts (Dict[str, str]): A dictionary of generated model artifacts and their paths.
        analysis (str): A textual analysis or summary of the solution's performance.
    """

    # General attributes
    id: str = field(default_factory=lambda: uuid.uuid4().hex, kw_only=True)
    created_time: float = field(default_factory=lambda: time.time(), kw_only=True)
    visited: bool = field(default=False, kw_only=True)
    depth: int = field(default=0, kw_only=True)

    # Directed edges to/from other nodes
    edges_in: List[Edge] = field(default_factory=list, hash=True, kw_only=True)
    edges_out: List[Edge] = field(default_factory=list, hash=True, kw_only=True)

    # Pre-execution contents: the solution plan and the generated code
    solution_plan: str
    training_code: str = field(default=None, hash=True, kw_only=True)
    inference_code: str = field(default=None, hash=True, kw_only=True)
    training_tests: str = field(default=None, hash=True, kw_only=True)
    inference_tests: str = field(default=None, hash=True, kw_only=True)

    # Pre-execution estimates of the node's value/cost, can be used to guide search
    estimated_value: float = field(default=None, kw_only=True)
    estimated_cost: float = field(default=None, kw_only=True)

    # Post-execution results: model performance, execution time, exceptions, etc.
    performance: Metric = field(default=None, kw_only=True)  # Validation performance
    test_performance: Metric = field(default=None, kw_only=True)  # Test set performance
    execution_time: float = field(default=None, kw_only=True)
    execution_stdout: list[str] = field(default_factory=list, kw_only=True)
    exception_was_raised: bool = field(default=False, kw_only=True)
    exception: Exception = field(default=None, kw_only=True)
    model_artifacts: List[Path] = field(default_factory=list, kw_only=True)
    analysis: str = field(default=None, kw_only=True)

    @property
    def is_terminal(self) -> bool:
        """
        Check if the node is terminal (i.e., has no outgoing edges).

        :return: [bool] True if the node has no outgoing edges, False otherwise.
        """
        return len(self.edges_out) == 0

    @property
    def is_root(self) -> bool:
        """
        Check if the node is a root (i.e., has no incoming edges).

        :return: [bool] True if the node has no incoming edges, False otherwise.
        """
        return len(self.edges_in) == 0
