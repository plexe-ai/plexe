"""
Module: random_policy

This module defines the `RandomSearchPolicy` class, which implements a search policy
that selects nodes randomly from a graph.

Classes:
    - RandomSearchPolicy: A search policy that selects nodes randomly from a graph.

Dependencies:
    - Graph: Represents a directed graph structure consisting of nodes and edges.
    - Node: Represents a node in the graph, with attributes for managing connections
      and execution results.
    - SearchPolicy: Abstract base class for defining search policies on a graph.

Example Usage:
    >>> from plexe.internal.models.entities.graph import Graph
    >>> from plexe.internal.models.search.random_policy import RandomSearchPolicy
    >>> graph = Graph()
    >>> policy = RandomSearchPolicy(graph)
    >>> selected_node = policy.select_node_enter()
"""

import random
from typing import List

from plexe.internal.models.entities.graph import Graph
from plexe.internal.models.entities.node import Node
from plexe.internal.models.search.policy import SearchPolicy


class RandomSearchPolicy(SearchPolicy):
    """
    A search policy that selects nodes randomly from a graph.
    """

    def __init__(self, graph: Graph):
        """
        Initialize the RandomSearchPolicy with a graph.

        :param graph: The graph to search.
        """
        super().__init__(graph)

    def select_node_enter(self, n: int = 1) -> List[Node]:
        """
        Select a node to enter randomly from the graph.

        :param n: The number of nodes to select. Currently, only 1 is supported.
        :return: A list containing one randomly selected node.
        :raises NotImplementedError: If n is not 1.
        """
        # n must be a positive integer less than the number of unvisited nodes
        if not 0 < n <= len(self.graph.unvisited_nodes):
            raise ValueError(f"Cannot select {n} nodes for testing from {len(self.graph.unvisited_nodes)} available.")
        # if there are no unvisited nodes, return the first node
        if not self.graph.unvisited_nodes:
            return [self.graph.nodes[0]] if self.graph.nodes else []
        # else return a random sample of unvisited nodes
        return random.sample(self.graph.unvisited_nodes, min(n, len(self.graph.unvisited_nodes)))

    def select_node_expand(self, n: int = 1) -> List[Node]:
        """
        Select a node to expand randomly from the graph.

        :param n: The number of nodes to select. Currently, only 1 is supported.
        :return: A list containing one randomly selected node.
        :raises NotImplementedError: If n is not 1.
        """
        # n must be a positive integer less than the number of available nodes
        if not 0 < n <= len(self.graph.nodes):
            raise ValueError(f"Cannot select {n} nodes for expansion from {len(self.graph.nodes)} available.")
        # Prefer to expand visited good nodes, then visited buggy nodes
        nodes = []
        if self.graph.good_nodes:
            nodes.extend(random.sample(self.graph.good_nodes, min(n, len(self.graph.good_nodes))))
        if len(nodes) < n and self.graph.buggy_nodes:
            nodes.extend(random.sample(self.graph.buggy_nodes, min(n - len(nodes), len(self.graph.buggy_nodes))))
        # If no nodes have been visited, return the first node
        if len(nodes) == 0:
            nodes.append(self.graph.nodes[0])
        return nodes
