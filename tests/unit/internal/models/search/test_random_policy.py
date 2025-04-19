# plexe/internal/models/search/test_random_policy.py:

"""
Unit tests for the RandomSearchPolicy class.

Tested Scenarios:
- Random node selection (single and multiple).
- Handling of invalid `n` values (e.g., too many, zero, or negative).
- Behaviour with an empty graph.
- Exclusion of buggy nodes from selection.

Dependencies:
- pytest: Used as the testing framework.
- Fixtures: `setup_graph` to create a sample graph and `random_search_policy`
  to initialise the policy.

"""

import pytest

from plexe.internal.models.entities.graph import Graph
from plexe.internal.models.entities.node import Node
from plexe.internal.models.search.random_policy import RandomSearchPolicy


@pytest.fixture
def setup_graph():
    """
    Fixture to set up a sample graph with nodes for testing purposes.
    """
    graph = Graph()

    # Create nodes
    node1 = Node(solution_plan="Plan A", training_code="code1", inference_code="code2", training_tests="code3")
    node2 = Node(solution_plan="Plan B", training_code="code4", inference_code="code5", training_tests="code6")
    node3 = Node(solution_plan="Plan C", training_code="code7", inference_code="code8", training_tests="code9")
    node4 = Node(solution_plan="Plan D", training_code="code10", inference_code="code11", training_tests="code12")

    # Add nodes to the graph
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)
    graph.add_node(node4)

    # Mark some nodes as buggy
    node2.visited = True
    node2.exception_was_raised = True
    # Mark some nodes as good
    node4.visited = True

    return graph


@pytest.fixture
def random_search_policy(setup_graph):
    """
    Fixture to initialize the RandomSearchPolicy with a sample graph.
    """
    return RandomSearchPolicy(setup_graph)


def test_select_node_enter_single(random_search_policy, setup_graph):
    """
    Test that select_node_enter selects a single random node from the graph.
    """
    selected_nodes = random_search_policy.select_node_enter(n=1)
    assert len(selected_nodes) == 1
    assert selected_nodes[0] in setup_graph.unvisited_nodes


def test_select_node_enter_multiple(random_search_policy, setup_graph):
    """
    Test that select_node_enter can select multiple nodes if n > 1.
    """
    selected_nodes = random_search_policy.select_node_enter(n=2)
    assert len(selected_nodes) == 2
    for node in selected_nodes:
        assert node in setup_graph.unvisited_nodes


def test_select_node_enter_too_many(random_search_policy):
    """
    Test that selecting more nodes than available raises an error.
    """
    with pytest.raises(ValueError):
        random_search_policy.select_node_enter(n=10)


def test_select_node_enter_zero_or_negative(random_search_policy):
    """
    Test that selecting zero or negative number of nodes raises an error.
    """
    with pytest.raises(ValueError):
        random_search_policy.select_node_enter(n=0)
    with pytest.raises(ValueError):
        random_search_policy.select_node_enter(n=-1)


def test_select_node_expand_single(random_search_policy, setup_graph):
    """
    Test that select_node_expand selects a single random node.
    """
    selected_nodes = random_search_policy.select_node_expand(n=1)
    assert len(selected_nodes) == 1
    assert selected_nodes[0] in setup_graph.good_nodes


def test_empty_graph():
    """
    Test that selecting a node from an empty graph raises an error.
    """
    empty_graph = Graph()
    empty_policy = RandomSearchPolicy(empty_graph)
    with pytest.raises(ValueError):
        empty_policy.select_node_enter()
