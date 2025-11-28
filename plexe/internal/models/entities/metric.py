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
from typing import Optional
from weakref import WeakValueDictionary


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


# Internal cache for sharing MetricComparator instances across all metrics
# This ensures only one comparator object exists per unique (method, target, epsilon) combination
_comparator_cache: WeakValueDictionary = WeakValueDictionary()


def _get_shared_comparator(comparison_method: ComparisonMethod, target: Optional[float] = None, epsilon: float = 1e-9) -> MetricComparator:
    """
    Get or create a shared MetricComparator instance.
    
    This function ensures that identical comparators are reused across all Metric instances,
    reducing memory usage and ensuring consistency.
    
    :param comparison_method: The comparison method.
    :param target: Optional target value for TARGET_IS_BETTER.
    :param epsilon: Tolerance for floating-point comparisons.
    :return: A shared MetricComparator instance.
    """
    # Create a cache key from the comparator parameters
    cache_key = (comparison_method, target, epsilon)
    
    # Try to get existing comparator from cache
    if cache_key in _comparator_cache:
        return _comparator_cache[cache_key]
    
    # Create new comparator and cache it
    comparator = MetricComparator(comparison_method, target, epsilon)
    _comparator_cache[cache_key] = comparator
    return comparator


class _MetricDefinition:
    """
    Internal class representing a metric type definition.
    
    This separates the metric definition (what it is) from the metric value (a measurement).
    Metric definitions are immutable and can be shared across multiple metric values.
    
    This is an internal implementation detail - users should not interact with this class directly.
    """
    
    def __init__(self, name: str, comparator: MetricComparator):
        """
        Initialize a metric definition.
        
        :param name: The name of the metric.
        :param comparator: The shared comparator instance.
        """
        self._name = name
        self._comparator = comparator
    
    @property
    def name(self) -> str:
        """The name of the metric."""
        return self._name
    
    @property
    def comparator(self) -> MetricComparator:
        """The shared comparator instance."""
        return self._comparator
    
    def __eq__(self, other) -> bool:
        """Check if two metric definitions are equal."""
        if not isinstance(other, _MetricDefinition):
            return False
        return (
            self.name == other.name
            and self.comparator.comparison_method == other.comparator.comparison_method
            and self.comparator.target == other.comparator.target
            and self.comparator.epsilon == other.comparator.epsilon
        )
    
    def __hash__(self) -> int:
        """Hash the metric definition."""
        return hash((self.name, self.comparator.comparison_method, self.comparator.target, self.comparator.epsilon))


@total_ordering
class Metric:
    """
    Represents a metric with a name, a value, and a comparator for determining which metric is better.

    This class internally separates the metric definition (type) from the metric value (measurement),
    and automatically shares comparator instances to reduce memory usage.

    Attributes:
        name (str): The name of the metric (e.g., 'accuracy', 'loss').
        value (float): The numeric value of the metric.
        comparator (MetricComparator): The comparison logic for the metric (shared instance).
    """

    def __init__(self, name: str, value: float = None, comparator: MetricComparator = None, is_worst: bool = False):
        """
        Initializes a Metric object.

        The comparator instance is automatically shared with other metrics that have the same
        comparison method, target, and epsilon values, reducing memory usage.

        :param name: The name of the metric.
        :param value: The numeric value of the metric.
        :param comparator: An instance of MetricComparator for comparison logic.
        :param is_worst: Indicates if the metric value is the worst possible value.
        """
        # Store the metric value (dynamic, instance-specific)
        self.value = value
        self.is_worst = is_worst or value is None
        
        # Get or create a shared comparator instance
        if comparator is not None:
            # Use the shared comparator cache to ensure we reuse identical comparators
            # This is the key optimization: identical comparators are shared across all metrics
            shared_comparator = _get_shared_comparator(
                comparison_method=comparator.comparison_method,
                target=comparator.target,
                epsilon=comparator.epsilon
            )
        else:
            # If no comparator provided, raise an error as it's required for a valid metric
            # This maintains the same behavior as before
            raise ValueError("Metric requires a comparator. Provide a MetricComparator instance.")
        
        # Create internal metric definition (separates type from value)
        # This is the key separation: definition (what it is) vs value (measurement)
        self._definition = _MetricDefinition(name=name, comparator=shared_comparator)
    
    @property
    def name(self) -> str:
        """The name of the metric (for backward compatibility)."""
        return self._definition.name
    
    @property
    def comparator(self) -> MetricComparator:
        """The shared comparator instance (for backward compatibility)."""
        return self._definition.comparator

    def __gt__(self, other) -> bool:
        """
        Determine if this metric is better than another metric.

        :param other: Another Metric object to compare against.
        :return: True if this metric is better, False otherwise.
        :raises ValueError: If the metrics have different names or comparison methods.
        """
        if not isinstance(other, Metric):
            return NotImplemented

        if self.is_worst:
            return False

        if other.is_worst:
            return True

        # Compare using definitions - this is cleaner and ensures consistency
        if self._definition != other._definition:
            # Provide detailed error message for backward compatibility
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

        # Use definition equality for cleaner comparison
        return (
            self._definition == other._definition
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
