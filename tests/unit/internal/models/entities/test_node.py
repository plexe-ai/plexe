"""
Unit tests for the Node and Edge classes.

These tests verify the core functionality of the Node and Edge classes, including:
- Initialization of Node and Edge instances.
- Relationships between nodes and edges (e.g., incoming and outgoing edges).
- Node properties such as `is_terminal` and `is_root`.
- Execution-related attributes and their default values.
- Unique ID generation for each Node.

The test suite is designed to prevent regressions and ensure consistent behaviour
as the library evolves.
"""

from plexe.internal.models.entities.node import Node, Edge


def test_node_initialization():
    node = Node(
        solution_plan="Plan A",
        training_code="print('training code')",
        inference_code="print('inference code')",
        training_tests="print('tests code')",
        estimated_value=100.0,
        estimated_cost=10.0,
        model_artifacts=["/path/to/model.pkl"],
    )

    assert node.id is not None
    assert isinstance(node.id, str)
    assert node.solution_plan == "Plan A"
    assert node.training_code == "print('training code')"
    assert node.inference_code == "print('inference code')"
    assert node.training_tests == "print('tests code')"
    assert node.estimated_value == 100.0
    assert node.estimated_cost == 10.0
    assert node.model_artifacts == ["/path/to/model.pkl"]
    assert node.edges_in == []
    assert node.edges_out == []


def test_edge_initialization():
    node1 = Node(
        solution_plan="Plan A",
        training_code="print('training code')",
        inference_code="print('inference code')",
        training_tests="print('tests code')",
    )
    node2 = Node(
        solution_plan="Plan B",
        training_code="print('other training code')",
        inference_code="print('other inference code')",
        training_tests="print('tests code')",
    )

    edge = Edge(source=node1, target=node2)
    node1.edges_out.append(edge)
    node2.edges_in.append(edge)

    assert edge.source == node1
    assert edge.target == node2
    assert edge in node1.edges_out
    assert edge in node2.edges_in


def test_node_is_terminal():
    node = Node(
        solution_plan="Plan A",
        training_code="print('training code')",
        inference_code="print('inference code')",
        training_tests="print('tests code')",
    )

    assert node.is_terminal  # Should be terminal since no outgoing edges

    node2 = Node(
        solution_plan="Plan B",
        training_code="print('other training code')",
        inference_code="print('other inference code')",
        training_tests="print('other tests code')",
    )
    edge = Edge(source=node, target=node2)
    node.edges_out.append(edge)

    assert not node.is_terminal  # Not terminal anymore

    node.edges_out.remove(edge)
    assert node.is_terminal  # Should be terminal again


def test_node_is_root():
    node = Node(
        solution_plan="Plan A",
        training_code="print('training code')",
        inference_code="print('inference code')",
        training_tests="print('tests code')",
    )

    assert node.is_root  # Should be root since no incoming edges

    node2 = Node(
        solution_plan="Plan B",
        training_code="print('other training code')",
        inference_code="print('other inference code')",
        training_tests="print('tests code')",
    )
    edge = Edge(source=node2, target=node)
    node.edges_in.append(edge)

    assert not node.is_root  # Not root anymore

    node.edges_in.remove(edge)
    assert node.is_root  # Should be root again


def test_node_execution_fields():
    node = Node(
        solution_plan="Plan A",
        training_code="print('training code')",
        inference_code="print('inference code')",
        training_tests="print('tests code')",
    )

    assert node.performance is None
    assert node.execution_stdout == []
    assert node.execution_time is None
    assert node.analysis is None
    assert node.exception_was_raised is False
    assert node.exception is None


def test_uuid_uniqueness():
    node1 = Node(
        solution_plan="Plan A",
        training_code="print('training code')",
        inference_code="print('inference code')",
        training_tests="print('tests code')",
    )
    node2 = Node(
        solution_plan="Plan B",
        training_code="print('other training code')",
        inference_code="print('other inference code')",
        training_tests="print('tests code')",
    )

    assert node1.id != node2.id
