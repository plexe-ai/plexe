# plexe/internal/models/search/best_first_policy.py

"""
This module defines the `BestFirstSearchPolicy` class, which implements a greedy search
policy that naively always selects the highest-scoring node from a graph first.

Classes:
    - BestFirstSearchPolicy: A search policy that selects nodes greedily from a graph.

Dependencies:
    - Graph: Represents a directed graph structure consisting of nodes and edges.
    - Node: Represents a node in the graph, with attributes for managing connections
      and execution results.
    - SearchPolicy: Abstract base class for defining search policies on a graph.

Example Usage:
    >>> from plexe.internal.models.entities.graph import Graph
    >>> from plexe.internal.models.search.best_first_policy import BestFirstSearchPolicy
    >>> graph = Graph()
    >>> policy = BestFirstSearchPolicy(graph)
    >>> selected_node = policy.select_node_enter()
"""

from typing import List

from plexe.internal.models.entities.graph import Graph
from plexe.internal.models.entities.metric import Metric
from plexe.internal.models.entities.node import Node
from plexe.internal.models.search.policy import SearchPolicy


class BestFirstSearchPolicy(SearchPolicy):
    """
    A search policy that greedily always selects the best node from a graph.
    """

    def __init__(self, graph: Graph):
        """
        Initialize the BestFirstSearchPolicy with a graph.

        :param graph: The graph to search.
        """
        super().__init__(graph)

    def select_node_enter(self, n: int = 1) -> List[Node]:
        """
        Select the best node(s) to enter from the graph.

        :param n: The number of nodes to select.
        :return: A list containing the top N nodes.
        """

        # todo: could this recursive graph traversal become a performance issue?
        # the graph size is not expected to be large, so it should be fine
        def get_closest_ancestor_performance(node: Node) -> Metric | None:
            if node.performance is not None:
                return node.performance
            elif node.edges_in:
                return max(get_closest_ancestor_performance(edge.source) for edge in node.edges_in)
            else:
                return Metric("dummy", is_worst=True)

        # n must be a positive integer less than the number of unvisited nodes
        if not 0 < n <= len(self.graph.unvisited_nodes):
            raise ValueError(f"Cannot select {n} nodes for testing from {len(self.graph.unvisited_nodes)} available.")
        # if there are no unvisited nodes, return the first node
        if not self.graph.unvisited_nodes:
            return [self.graph.nodes[0]] if self.graph.nodes else []
        # Prefer nodes with ancestors that have higher performance
        return sorted(
            self.graph.unvisited_nodes, key=lambda node: get_closest_ancestor_performance(node), reverse=True
        )[:n]

    def select_node_expand(self, n: int = 1) -> List[Node]:
        """
        Select the best node(s) to expand from the graph.

        :param n: The number of nodes to select.
        :return: A list containing the top N nodes to expand.
        """
        # n must be a positive integer less than the number of available nodes
        if not 0 < n <= len(self.graph.nodes):
            raise ValueError(f"Cannot select {n} nodes for expansion from {len(self.graph.nodes)} available.")
        # Prefer to expand visited good nodes, then visited buggy nodes
        nodes = []
        if self.graph.good_nodes:
            nodes.extend(
                # Prefer good nodes with higher performance
                sorted(self.graph.good_nodes, key=lambda node: node.performance, reverse=True)[
                    : min(n, len(self.graph.good_nodes))
                ]  # Take as many top nodes as possible
            )
        if len(nodes) < n and self.graph.buggy_nodes:
            nodes.extend(
                # Prefer buggy nodes with fewer edges out (i.e., less explored)
                sorted(self.graph.buggy_nodes, key=lambda node: len(node.edges_out))[
                    : min(n, len(self.graph.buggy_nodes))
                ]  # Take as many top nodes as possible
            )
        # If no nodes have been visited, return the first node
        if len(nodes) == 0:
            nodes.append(self.graph.nodes[0])
        return nodes
