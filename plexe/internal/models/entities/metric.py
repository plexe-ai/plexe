"""
Module: plexe/internal/common/dataclasses/metric

This module defines classes for handling and comparing metrics in a flexible and extensible way.

Classes:
    - ComparisonMethod: Enum defining methods for comparing metrics.
    - MetricComparator: Encapsulates comparison logic for metrics, including methods like higher-is-better,
      lower-is-better, and target-is-better.
    - Metric: Represents a specific metric with a name, value, and comparator, allowing metrics to be compared
      and evaluated.

Example Usage:
    from metric_class import Metric, MetricComparator, ComparisonMethod

    comparator = MetricComparator(ComparisonMethod.HIGHER_IS_BETTER)
    metric1 = Metric(name="accuracy", value=0.8, comparator=comparator)
    metric2 = Metric(name="accuracy", value=0.9, comparator=comparator)

    print(metric1 < metric2)  # True
"""

from enum import Enum
from functools import total_ordering


class ComparisonMethod(Enum):
    """
    Defines methods for comparing metrics.

    Attributes:
        HIGHER_IS_BETTER: Indicates that higher values are better.
        LOWER_IS_BETTER: Indicates that lower values are better.
        TARGET_IS_BETTER: Indicates that values closer to a target are better.
    """

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    TARGET_IS_BETTER = "target_is_better"


class MetricComparator:
    """
    Encapsulates comparison logic for metrics.

    Attributes:
        comparison_method (ComparisonMethod): The method used to compare metrics.
        target (float, optional): The target value for TARGET_IS_BETTER comparisons.
    """

    def __init__(self, comparison_method: ComparisonMethod, target: float = None, epsilon: float = 1e-9):
        """
        Initializes the MetricComparator.

        :param comparison_method: The method to compare metric values.
        :param target: The target value for TARGET_IS_BETTER comparisons (optional).
        :param epsilon: The tolerance for floating-point error in TARGET_IS_BETTER comparisons (default: 1e-9).
        :raises ValueError: If TARGET_IS_BETTER is used without a target value.
        """
        self.comparison_method = comparison_method
        self.target = target if comparison_method == ComparisonMethod.TARGET_IS_BETTER else None
        self.epsilon = epsilon

        if self.comparison_method == ComparisonMethod.TARGET_IS_BETTER and self.target is None:
            raise ValueError("'TARGET_IS_BETTER' comparison requires a target value.")
        if self.comparison_method == ComparisonMethod.TARGET_IS_BETTER and not isinstance(self.target, (float, int)):
            raise ValueError("'TARGET_IS_BETTER' requires a numeric target value.")

    def compare(self, value1: float, value2: float) -> int:
        """
        Compare two metric values based on the defined comparison method.

        :param value1: The first metric value.
        :param value2: The second metric value.
        :return: -1 if value1 is better, 1 if value2 is better, 0 if they are equal.
        :raises ValueError: If an invalid comparison method is used.
        """
        if value1 is None and value2 is None:
            return 0
        elif value1 is None:
            return 1
        elif value2 is None:
            return -1
        elif self.comparison_method == ComparisonMethod.HIGHER_IS_BETTER:
            return (value2 > value1 + self.epsilon) - (value1 > value2 + self.epsilon)
        elif self.comparison_method == ComparisonMethod.LOWER_IS_BETTER:
            return (value1 > value2 + self.epsilon) - (value2 > value1 + self.epsilon)
        elif self.comparison_method == ComparisonMethod.TARGET_IS_BETTER:
            dist1 = abs(value1 - self.target)
            dist2 = abs(value2 - self.target)
            if dist1 > dist2 + self.epsilon:
                return 1
            elif dist2 > dist1 + self.epsilon:
                return -1
            else:
                return 0
        else:
            raise ValueError("Invalid comparison method.")


# todo: this class is a mess as it mixes concerns of a metric and a metric value; needs refactoring
@total_ordering
class Metric:
    """
    Represents a metric with a name, a value, and a comparator for determining which metric is better.

    Attributes:
        name (str): The name of the metric (e.g., 'accuracy', 'loss').
        value (float): The numeric value of the metric.
        comparator (MetricComparator): The comparison logic for the metric.
    """

    def __init__(self, name: str, value: float = None, comparator: MetricComparator = None, is_worst: bool = False):
        """
        Initializes a Metric object.

        :param name: The name of the metric.
        :param value: The numeric value of the metric.
        :param comparator: An instance of MetricComparator for comparison logic.
        :param is_worst: Indicates if the metric value is the worst possible value.
        """
        self.name = name
        self.value = value
        self.comparator = comparator
        self.is_worst = is_worst or value is None

    def __gt__(self, other) -> bool:
        """
        Determine if this metric is better than another metric.

        :param other: Another Metric object to compare against.
        :return: True if this metric is better, False otherwise.
        :raises ValueError: If the metrics have different names or comparison methods.
        """
        if not isinstance(other, Metric):
            return NotImplemented

        if self.is_worst or (self.is_worst and other.is_worst):
            return False

        if other.is_worst:
            return True

        if self.name != other.name:
            raise ValueError("Cannot compare metrics with different names.")

        if self.comparator.comparison_method != other.comparator.comparison_method:
            raise ValueError("Cannot compare metrics with different comparison methods.")

        if (
            self.comparator.comparison_method == ComparisonMethod.TARGET_IS_BETTER
            and self.comparator.target != other.comparator.target
        ):
            raise ValueError("Cannot compare 'TARGET_IS_BETTER' metrics with different target values.")

        return self.comparator.compare(self.value, other.value) < 0

    def __eq__(self, other) -> bool:
        """
        Check if this metric is equal to another metric.

        :param other: Another Metric object to compare against.
        :return: True if the metrics are equal, False otherwise.
        """
        if not isinstance(other, Metric):
            return NotImplemented

        if self.is_worst and other.is_worst:
            return True

        if self.is_worst or other.is_worst:
            return False

        return (
            self.name == other.name
            and self.comparator.comparison_method == other.comparator.comparison_method
            and self.comparator.compare(self.value, other.value) == 0
        )

    def __repr__(self) -> str:
        """
        Return a string representation of the Metric object.

        :return: A string representation of the Metric.
        """
        target_str = (
            f", target={self.comparator.target}"
            if self.comparator.comparison_method == ComparisonMethod.TARGET_IS_BETTER
            else ""
        )
        return f"Metric(name={self.name!r}, value={self.value}, comparison={self.comparator.comparison_method.name}{target_str})"

    def __str__(self) -> str:
        """
        Return a user-friendly string representation of the Metric.

        :return: A string describing the Metric.
        """
        comparison_symbols = {
            ComparisonMethod.HIGHER_IS_BETTER: "↑",
            ComparisonMethod.LOWER_IS_BETTER: "↓",
            ComparisonMethod.TARGET_IS_BETTER: "≈",
        }
        symbol = comparison_symbols.get(self.comparator.comparison_method, "?")
        return f"Metric {self.name} {symbol} {self.value}"

    @property
    def is_valid(self) -> bool:
        """
        Check if the metric value is valid (i.e., not None or NaN).

        :return: True if the metric value is valid, False otherwise.
        """
        return self.value is not None and not (self.value != self.value)  # NaN check
