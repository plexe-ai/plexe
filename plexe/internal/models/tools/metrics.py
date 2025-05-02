"""
Tools related to metrics selection and model review/metadata extraction.
"""

import logging
from typing import Dict, Callable

from smolagents import tool

from plexe.internal.common.provider import Provider
from plexe.internal.models.generation.planning import SolutionPlanGenerator

logger = logging.getLogger(__name__)


def get_select_target_metric(llm_to_use: str) -> Callable:
    """Returns a tool function to select target metrics with the model ID pre-filled."""

    @tool
    def select_target_metric(task: str) -> Dict:
        """
        Selects the appropriate target metric to optimise for the given task.

        Args:
            task: The task definition combining intent, input schema, and output schema

        Returns:
            A dictionary containing the metric information
        """
        plan_generator = SolutionPlanGenerator(Provider(llm_to_use))
        metric = plan_generator.select_target_metric(task)
        return {
            "name": metric.name,
            "value": metric.value,
            "comparison_method": str(metric.comparator.comparison_method),
        }

    return select_target_metric
