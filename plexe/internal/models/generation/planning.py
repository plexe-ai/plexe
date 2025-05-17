# plexe/internal/models/generation/planning.py

"""
This module provides functions and classes for generating and planning solutions for machine learning problems.
"""

import json
import logging

from pydantic import BaseModel

from plexe.config import prompt_templates
from plexe.internal.common.provider import Provider
from plexe.internal.models.entities.metric import Metric, MetricComparator, ComparisonMethod

logger = logging.getLogger(__name__)


class SolutionPlanGenerator:
    """
    A class to generate solution plans for given problem statements.
    """

    def __init__(self, provider: Provider):
        """
        Initializes the SolutionPlanGenerator with an empty context.
        """
        self.provider: Provider = provider

    def select_target_metric(self, problem_statement: str) -> Metric:
        """
        Selects the metric to optimise for the given problem statement and dataset.

        :param problem_statement: definition of the problem
        :return: the metric to optimise
        """

        class MetricResponse(BaseModel):
            name: str
            comparison_method: ComparisonMethod
            comparison_target: float = None

        response: MetricResponse = MetricResponse(
            **json.loads(
                self.provider.query(
                    system_message=prompt_templates.planning_system(),
                    user_message=prompt_templates.planning_select_metric(
                        problem_statement=problem_statement,
                    ),
                    response_format=MetricResponse,
                )
            )
        )

        try:
            return Metric(
                name=response.name,
                value=float("inf") if response.comparison_method == ComparisonMethod.LOWER_IS_BETTER else -float("inf"),
                comparator=MetricComparator(response.comparison_method, response.comparison_target),
            )
        except Exception as e:
            raise ValueError(f"Could not determine optimization metric from problem statement: {response}") from e
