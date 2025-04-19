"""
Module: test_metric_class

This module contains unit tests for the `Metric` and `MetricComparator` classes, ensuring their functionality
and robustness. The tests cover:

- Comparison methods (`HIGHER_IS_BETTER`, `LOWER_IS_BETTER`, `TARGET_IS_BETTER`).
- Handling of edge cases like floating-point precision and boundary values.
- Validation logic for metrics (e.g., checking valid/invalid states).
- Compatibility and error handling for metrics with different names or comparison methods.
- System-level behaviours, such as sorting collections of metrics.

Dependencies:
    - pytest: For running the test suite.
    - metric_class: The module containing the `Metric` and `MetricComparator` class implementations.

Example:
    pytest test_metric_class.py
"""

import pytest
from plexe.internal.models.entities.metric import Metric, MetricComparator, ComparisonMethod


def test_comparator_higher_is_better():
    comparator = MetricComparator(ComparisonMethod.HIGHER_IS_BETTER)
    assert comparator.compare(0.8, 0.9) == 1  # 0.9 is better
    assert comparator.compare(0.9, 0.8) == -1  # 0.9 is better
    assert comparator.compare(0.8, 0.8) == 0  # Equal


def test_comparator_lower_is_better():
    comparator = MetricComparator(ComparisonMethod.LOWER_IS_BETTER)
    assert comparator.compare(0.8, 0.9) == -1  # 0.8 is better
    assert comparator.compare(0.9, 0.8) == 1  # 0.8 is better
    assert comparator.compare(0.8, 0.8) == 0  # Equal


def test_comparator_target_is_better():
    comparator = MetricComparator(ComparisonMethod.TARGET_IS_BETTER, target=1.0)
    assert comparator.compare(0.9, 1.1) == 0  # Both are equally close to the target
    assert comparator.compare(1.0, 1.2) == -1  # 1.0 is closer to the target
    assert comparator.compare(0.8, 1.0) == 1  # 1.0 is closer to the target


def test_comparator_invalid_target():
    with pytest.raises(ValueError, match="requires a target value"):
        MetricComparator(ComparisonMethod.TARGET_IS_BETTER)


def test_comparator_floating_point_precision():
    comparator = MetricComparator(ComparisonMethod.HIGHER_IS_BETTER)
    assert comparator.compare(1.0000001, 1.0000002) == 1
    assert comparator.compare(1.0000002, 1.0000001) == -1


def test_metric_higher_is_better():
    comparator = MetricComparator(ComparisonMethod.HIGHER_IS_BETTER)
    metric1 = Metric(name="accuracy", value=0.8, comparator=comparator)
    metric2 = Metric(name="accuracy", value=0.9, comparator=comparator)
    assert metric1 < metric2
    assert metric2 > metric1
    assert metric1 != metric2


def test_metric_lower_is_better():
    comparator = MetricComparator(ComparisonMethod.LOWER_IS_BETTER)
    metric1 = Metric(name="loss", value=0.8, comparator=comparator)
    metric2 = Metric(name="loss", value=0.6, comparator=comparator)
    assert metric1 < metric2  # metric1 is "lower" because it's worse
    assert metric2 > metric1
    assert metric1 != metric2


def test_metric_target_is_better():
    comparator = MetricComparator(ComparisonMethod.TARGET_IS_BETTER, target=1.0)
    metric1 = Metric(name="value", value=0.9, comparator=comparator)
    metric2 = Metric(name="value", value=1.1, comparator=comparator)
    assert metric1 == metric2  # Both are equally close to the target


def test_metric_different_names():
    comparator = MetricComparator(ComparisonMethod.HIGHER_IS_BETTER)
    metric1 = Metric(name="accuracy", value=0.8, comparator=comparator)
    metric2 = Metric(name="loss", value=0.9, comparator=comparator)
    with pytest.raises(ValueError, match="Cannot compare metrics with different names"):
        metric1 > metric2


def test_metric_invalid_comparison():
    comparator1 = MetricComparator(ComparisonMethod.HIGHER_IS_BETTER)
    comparator2 = MetricComparator(ComparisonMethod.LOWER_IS_BETTER)
    comparator3 = MetricComparator(ComparisonMethod.TARGET_IS_BETTER, target=1.0)
    comparator4 = MetricComparator(ComparisonMethod.TARGET_IS_BETTER, target=2.0)
    metric1 = Metric(name="accuracy", value=0.8, comparator=comparator1)
    metric2 = Metric(name="accuracy", value=0.9, comparator=comparator2)
    metric3 = Metric(name="accuracy", value=1.0, comparator=comparator3)
    metric4 = Metric(name="accuracy", value=1.1, comparator=comparator4)
    with pytest.raises(ValueError, match="Cannot compare metrics with different comparison methods"):
        metric1 > metric2
    with pytest.raises(ValueError, match="Cannot compare 'TARGET_IS_BETTER' metrics with different target values"):
        metric3 > metric4


def test_metric_is_valid():
    comparator = MetricComparator(ComparisonMethod.HIGHER_IS_BETTER)
    metric = Metric(name="accuracy", value=float("nan"), comparator=comparator)
    assert not metric.is_valid

    metric = Metric(name="accuracy", value=0.8, comparator=comparator)
    assert metric.is_valid


def test_metric_repr_and_str():
    comparator = MetricComparator(ComparisonMethod.HIGHER_IS_BETTER)
    metric = Metric(name="accuracy", value=0.8, comparator=comparator)
    assert repr(metric) == "Metric(name='accuracy', value=0.8, comparison=HIGHER_IS_BETTER)"
    assert str(metric) == "Metric accuracy ↑ 0.8"


def test_metric_transitivity():
    comparator = MetricComparator(ComparisonMethod.HIGHER_IS_BETTER)
    metric1 = Metric(name="accuracy", value=0.8, comparator=comparator)
    metric2 = Metric(name="accuracy", value=0.9, comparator=comparator)
    metric3 = Metric(name="accuracy", value=1.0, comparator=comparator)
    assert metric1 < metric2 < metric3
    assert metric3 > metric2 > metric1


def test_metric_collection_sorting():
    comparator = MetricComparator(ComparisonMethod.HIGHER_IS_BETTER)
    metrics = [
        Metric(name="accuracy", value=0.8, comparator=comparator),
        Metric(name="accuracy", value=0.6, comparator=comparator),
        Metric(name="accuracy", value=0.9, comparator=comparator),
    ]
    metrics.sort(reverse=True)
    assert [m.value for m in metrics] == [0.9, 0.8, 0.6]
