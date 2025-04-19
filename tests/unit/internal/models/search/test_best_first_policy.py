# plexe/internal/models/search/test_best_first_policy.py:

"""
Unit tests for the BestFirstSearchPolicy class.

Tested Scenarios:
- Node selection (single and multiple).
- Handling of invalid `n` values (e.g., too many, zero, or negative).
- Behaviour with an empty graph.
- Exclusion of buggy nodes from selection.

Dependencies:
- pytest: Used as the testing framework.
- Fixtures: `setup_graph` to create a sample graph and `best_first_policy`
  to initialise the policy.

"""

import pytest

from plexe.internal.models.entities.graph import Graph
from plexe.internal.models.entities.metric import Metric, MetricComparator, ComparisonMethod
from plexe.internal.models.entities.node import Node
from plexe.internal.models.search.best_first_policy import BestFirstSearchPolicy


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
    node5 = Node(solution_plan="Plan E", training_code="code13", inference_code="code14", training_tests="code15")
    node6 = Node(solution_plan="Plan F", training_code="code16", inference_code="code17", training_tests="code18")
    node7 = Node(solution_plan="Plan G", training_code="code19", inference_code="code20", training_tests="code21")

    # Add nodes to the graph
    graph.add_node(node1)
    graph.add_node(node2, parent=node1)
    graph.add_node(node3, parent=node2)
    graph.add_node(node4, parent=node2)
    graph.add_node(node5, parent=node4)
    graph.add_node(node6, parent=node5)
    graph.add_node(node7, parent=node4)

    # Mark some nodes as buggy
    node1.visited = True
    node1.exception_was_raised = True
    node2.visited = True
    node2.exception_was_raised = True
    # Mark some nodes as good
    node4.visited = True
    node4.performance = Metric("testmetric", 0.9, comparator=MetricComparator(ComparisonMethod.HIGHER_IS_BETTER))
    node5.visited = True
    node5.performance = Metric("testmetric", 0.8, comparator=MetricComparator(ComparisonMethod.HIGHER_IS_BETTER))

    return graph


@pytest.fixture
def best_first_search_policy(setup_graph):
    """
    Fixture to initialize the BestFirstSearchPolicy with a sample graph.
    """
    return BestFirstSearchPolicy(setup_graph)


def test_select_node_enter_single(best_first_search_policy, setup_graph):
    """
    Test that select_node_enter selects a single node from the graph.
    """
    selected_nodes = best_first_search_policy.select_node_enter(n=1)
    assert len(selected_nodes) == 1
    assert selected_nodes[0] in setup_graph.unvisited_nodes


def test_select_node_enter_multiple(best_first_search_policy, setup_graph):
    """
    Test that select_node_enter can select multiple nodes if n > 1.
    """
    selected_nodes = best_first_search_policy.select_node_enter(n=2)
    assert len(selected_nodes) == 2
    for node in selected_nodes:
        assert node in setup_graph.unvisited_nodes


def test_select_node_enter_too_many(best_first_search_policy):
    """
    Test that selecting more nodes than available raises an error.
    """
    with pytest.raises(ValueError):
        best_first_search_policy.select_node_enter(n=10)


def test_select_node_enter_zero_or_negative(best_first_search_policy):
    """
    Test that selecting zero or negative number of nodes raises an error.
    """
    with pytest.raises(ValueError):
        best_first_search_policy.select_node_enter(n=0)
    with pytest.raises(ValueError):
        best_first_search_policy.select_node_enter(n=-1)


def test_select_node_enter_selects_highest_ancestor_performance(best_first_search_policy, setup_graph):
    """
    Test that select_node_enter selects the node with the highest ancestor performance.
    """
    selected_nodes = best_first_search_policy.select_node_enter(n=1)
    assert selected_nodes[0].solution_plan == "Plan G"  # Node 7 has the highest ancestor performance


def test_select_node_expand_single(best_first_search_policy, setup_graph):
    """
    Test that select_node_expand selects a single node.
    """
    selected_nodes = best_first_search_policy.select_node_expand(n=1)
    assert len(selected_nodes) == 1
    assert selected_nodes[0] in setup_graph.good_nodes


def test_select_node_expand_selects_highest_performance(best_first_search_policy, setup_graph):
    """
    Test that select_node_expand selects the node with the highest performance.
    """
    selected_nodes = best_first_search_policy.select_node_expand(n=1)
    assert selected_nodes[0].performance == max(node.performance for node in setup_graph.good_nodes)


def test_empty_graph():
    """
    Test that selecting a node from an empty graph raises an error.
    """
    empty_graph = Graph()
    empty_policy = BestFirstSearchPolicy(empty_graph)
    with pytest.raises(ValueError):
        empty_policy.select_node_enter()
