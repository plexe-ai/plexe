"""
Module: policy

This module defines the `SearchPolicy` abstract base class, which provides an interface
for implementing various search policies on a graph.

Classes:
    - SearchPolicy: Abstract base class for defining search policies on a graph.

Dependencies:
    - Graph: Represents a directed graph structure consisting of nodes and edges.
    - Node: Represents a node in the graph, with attributes for managing connections
      and execution results.

Example Usage:
    >>> from plexe.internal.models.entities.graph import Graph
    >>> from plexe.internal.models.search.policy import SearchPolicy
    >>> class MySearchPolicy(SearchPolicy):
    >>>     def select_node_enter(self, n: int = 1):
    >>>         pass
    >>>     def select_node_expand(self, n: int = 1):
    >>>         pass
    >>> graph = Graph()
    >>> policy = MySearchPolicy(graph)
"""

import abc
from typing import List

from plexe.internal.models.entities.graph import Graph
from plexe.internal.models.entities.node import Node


class SearchPolicy(abc.ABC):
    """
    Abstract base class for defining search policies on a graph.
    """

    @abc.abstractmethod
    def __init__(self, graph: Graph):
        """
        Initialize the search policy with a graph.

        :param graph: The graph on which the search policy will operate.
        """
        self.graph = graph

    @abc.abstractmethod
    def select_node_enter(self, n: int = 1) -> List[Node]:
        """
        Select nodes to enter the search.

        :param n: The number of nodes to select.
        :return: A list of selected nodes.
        """
        pass

    @abc.abstractmethod
    def select_node_expand(self, n: int = 1) -> List[Node]:
        """
        Select nodes to expand in the search.

        :param n: The number of nodes to select.
        :return: A list of selected nodes.
        """
        pass
