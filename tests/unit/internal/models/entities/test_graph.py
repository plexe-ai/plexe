"""
Module: test_graph_class

This module contains unit tests for the `Graph` class, which represents a directed graph structure
consisting of nodes and edges. The tests cover various functionalities such as:

- Graph initialisation.
- Adding nodes with and without parents.
- Ensuring proper relationships between nodes and edges.
- Querying buggy and good nodes in the graph.
- Handling edge cases like duplicate nodes, circular references, and disconnected subgraphs.
- Stress testing with large graphs.

Dependencies:
    - pytest: For running the test suite.
    - graph_class: The module containing the `Graph` class implementation.
    - Node, Edge: Classes representing graph components.
"""

from plexe.internal.models.entities.graph import Graph
from plexe.internal.models.entities.node import Node


def test_graph_initialisation():
    graph = Graph()
    assert len(graph.nodes) == 0
    assert len(graph.edges) == 0


def test_add_single_node():
    graph = Graph()
    node = Node(solution_plan="Plan A", training_code="code1", inference_code="code2", training_tests="code3")
    graph.add_node(node)
    assert len(graph.nodes) == 1
    assert graph.nodes[0] == node
    assert len(graph.edges) == 0


def test_add_node_with_parent():
    graph = Graph()
    parent = Node(solution_plan="Plan Parent", training_code="code1", inference_code="code2", training_tests="code3")
    child = Node(solution_plan="Plan Child", training_code="code4", inference_code="code5", training_tests="code6")
    graph.add_node(child, parent=parent)

    assert len(graph.nodes) == 2
    assert parent in graph.nodes
    assert child in graph.nodes

    assert len(graph.edges) == 1
    edge = graph.edges[0]
    assert edge.source == parent
    assert edge.target == child

    assert edge in parent.edges_out
    assert edge in child.edges_in


def test_add_duplicate_nodes():
    graph = Graph()
    node = Node(solution_plan="Plan A", training_code="code1", inference_code="code2", training_tests="code3")
    graph.add_node(node)
    graph.add_node(node)
    assert len(graph.nodes) == 1


def test_duplicate_node_with_different_parents():
    graph = Graph()
    parent1 = Node(solution_plan="Parent1", training_code="code1", inference_code="code2", training_tests="code3")
    parent2 = Node(solution_plan="Parent2", training_code="code4", inference_code="code5", training_tests="code6")
    child = Node(solution_plan="Child", training_code="code7", inference_code="code8", training_tests="code9")

    graph.add_node(child, parent=parent1)
    graph.add_node(child, parent=parent2)

    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2


def test_edges_in_out():
    graph = Graph()
    parent = Node(solution_plan="Plan Parent", training_code="code1", inference_code="code2", training_tests="code3")
    child = Node(solution_plan="Plan Child", training_code="code4", inference_code="code5", training_tests="code6")
    graph.add_node(child, parent=parent)

    edge = graph.edges[0]
    assert edge in parent.edges_out
    assert edge in child.edges_in


def test_circular_reference():
    graph = Graph()
    node = Node(solution_plan="Circular", training_code="code1", inference_code="code2", training_tests="code3")
    graph.add_node(node, parent=node)
    assert len(graph.nodes) == 1
    assert len(graph.edges) == 1
    assert graph.edges[0].source == node
    assert graph.edges[0].target == node


def test_disconnected_subgraphs():
    graph = Graph()
    node1 = Node(solution_plan="Subgraph1", training_code="code1", inference_code="code2", training_tests="code3")
    node2 = Node(solution_plan="Subgraph2", training_code="code4", inference_code="code5", training_tests="code6")
    graph.add_node(node1)
    graph.add_node(node2)
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 0


def test_buggy_nodes():
    graph = Graph()
    node1 = Node(
        solution_plan="Plan A",
        training_code="code1",
        inference_code="code2",
        training_tests="code3",
        exception_was_raised=False,
        visited=True,
    )
    node2 = Node(
        solution_plan="Plan B",
        training_code="code4",
        inference_code="code5",
        training_tests="code6",
        exception_was_raised=True,
        visited=True,
    )

    graph.add_node(node1)
    graph.add_node(node2)

    buggy = graph.buggy_nodes
    assert len(buggy) == 1
    assert buggy[0] == node2


def test_good_nodes():
    graph = Graph()
    node1 = Node(
        solution_plan="Plan A",
        training_code="code1",
        inference_code="code2",
        training_tests="code3",
        exception_was_raised=False,
        visited=True,
    )
    node2 = Node(
        solution_plan="Plan B",
        training_code="code4",
        inference_code="code5",
        training_tests="code6",
        exception_was_raised=True,
        visited=True,
    )

    graph.add_node(node1)
    graph.add_node(node2)

    good = graph.good_nodes
    assert len(good) == 1
    assert good[0] == node1


def test_large_graph():
    graph = Graph()
    nodes = [
        Node(solution_plan=f"Node {i}", training_code="code", inference_code="code", training_tests="code")
        for i in range(1000)
    ]
    for i, node in enumerate(nodes):
        parent = nodes[i - 1] if i > 0 else None
        graph.add_node(node, parent=parent)
    assert len(graph.nodes) == 1000
    assert len(graph.edges) == 999
