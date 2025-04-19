"""
Tools related to metrics selection and model review/metadata extraction.
"""

import logging
from typing import Dict

from smolagents import tool

from plexe.internal.common.provider import Provider
from plexe.internal.models.generation.planning import SolutionPlanGenerator

logger = logging.getLogger(__name__)


@tool
def select_target_metric(task: str, provider: str) -> Dict:
    """
    Selects the appropriate target metric to optimise for the given task.

    Args:
        task: The task definition combining intent, input schema, and output schema
        provider: The string identifier representing the model that should be used, e.g. 'openai/gpt-4o'

    Returns:
        A dictionary containing the metric information
    """
    plan_generator = SolutionPlanGenerator(Provider(provider))
    metric = plan_generator.select_target_metric(task)
    return {"name": metric.name, "value": metric.value, "comparison_method": str(metric.comparator.comparison_method)}
