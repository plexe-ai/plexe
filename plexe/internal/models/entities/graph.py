"""
Module: graph_class

This module defines the `Graph` class, which represents a directed graph structure
consisting of nodes and edges. It provides functionality to manage the nodes and edges
of the graph while ensuring proper relationships between them.

Classes:
    - Graph: Represents a directed graph and provides methods to add nodes and query
      the graph's structure.

Dependencies:
    - Node: Represents a node in the graph, with attributes for managing connections
      and execution results.
    - Edge: Represents a directed edge between two nodes in the graph.

Example Usage:
    >>> from plexe.internal.models.entities.graphs.graph import Graph
    >>> graph = Graph()
    >>> node1 = Node(solution_plan="Plan A", training_code="code1", inference_code="code2", training_tests="code3")
    >>> node2 = Node(solution_plan="Plan B", training_code="code4", inference_code="code5", training_tests="code6")
    >>> graph.add_node(node1)
    >>> graph.add_node(node2, parent=node1)
    >>> print(graph.nodes)
    >>> print(graph.edges)
"""

from plexe.internal.models.entities.node import Node, Edge


class Graph:
    """
    Represents a directed graph structure consisting of nodes and edges.
    """

    def __init__(self):
        """
        Initializes an empty graph with no nodes or edges.
        """
        self._nodes = []
        self._edges = []

    @property
    def nodes(self) -> list[Node]:
        """
        Provides read-only access to the list of nodes.

        :return: The list of nodes in the graph.
        """
        return self._nodes

    @property
    def edges(self) -> list[Edge]:
        """
        Provides read-only access to the list of edges.

        :return: The list of edges in the graph.
        """
        return self._edges

    def add_node(self, node: Node, parent: Node = None) -> None:
        """
        Adds a node to the graph, with an optional inbound edge from the parent node.
        If a parent is provided, an edge is created between the parent and the new node.

        :param node: The node to add to the graph.
        :param parent: The parent node of the new node, if any.
        :return: None
        """
        if node not in self._nodes:
            self._nodes.append(node)
        if parent is not None:
            if parent not in self._nodes:
                self.add_node(parent)  # Ensure parent is added first
            edge = Edge(source=parent, target=node)
            if edge not in self._edges:
                self._edges.append(edge)
                parent.edges_out.append(edge)
                node.edges_in.append(edge)
        if parent is not None:
            node.depth = parent.depth + 1

    @property
    def buggy_nodes(self) -> list[Node]:
        """
        Returns a list of nodes that are flagged as buggy.

        :return: A list of nodes with exceptions raised during their execution.
        """
        return [n for n in self._nodes if n.exception_was_raised and n.visited]

    @property
    def good_nodes(self) -> list[Node]:
        """
        Returns a list of nodes that are not buggy (considered valid/good).

        :return: A list of nodes without exceptions raised during their execution.
        """
        return [n for n in self._nodes if not n.exception_was_raised and n.visited]

    @property
    def unvisited_nodes(self) -> list[Node]:
        """
        Returns a list of nodes that have not been visited.

        :return: A list of nodes that have not been visited.
        """
        return [n for n in self._nodes if not n.visited]
