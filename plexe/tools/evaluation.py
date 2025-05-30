"""
This module defines agent tools for evaluating the properties and performance of models.
"""

import logging
from typing import Dict, Callable

from smolagents import tool

from plexe.internal.common.provider import Provider
from plexe.internal.models.generation.review import ModelReviewer
from plexe.tools.schemas import get_solution_schemas

logger = logging.getLogger(__name__)


def get_review_finalised_model(llm_to_use: str) -> Callable:
    """Returns a tool function to review finalized models with the model ID pre-filled."""

    @tool
    def review_finalised_model(
        intent: str,
        solution_id: str,
    ) -> dict:
        """
        Reviews the entire model and extracts metadata. Use this function once you have completed work on the model, and
        you want to 'wrap up' the work by performing a holistic review of what has been built.

        Args:
            intent: The model intent
            solution_id: The solution ID to review

        Returns:
            A dictionary containing a summary and review of the model
        """
        from plexe.core.object_registry import ObjectRegistry
        from plexe.core.entities.solution import Solution

        object_registry = ObjectRegistry()

        try:
            schemas = get_solution_schemas(solution_id)
            input_schema = schemas["input"]
            output_schema = schemas["output"]
        except Exception:
            raise ValueError("Failed to retrieve schemas. Was schema resolution completed?")

        try:
            solution = object_registry.get(Solution, solution_id)
        except Exception:
            raise ValueError(f"Solution with ID '{solution_id}' not found. Was the solution created?")

        if not solution.training_code:
            raise ValueError("Training code not found in solution. Was the solution implemented?")

        if not solution.inference_code:
            raise ValueError("Inference code not found in solution. Was the inference code produced?")

        # Review the model using the ModelReviewer
        reviewer = ModelReviewer(Provider(llm_to_use))
        r = reviewer.review_model(
            intent, input_schema, output_schema, solution.plan, solution.training_code, solution.inference_code
        )

        # Update the solution with the review
        solution.review = r
        object_registry.register(Solution, solution_id, solution, overwrite=True)
        return r

    return review_finalised_model


@tool
def get_solution_performances() -> Dict[str, float]:
    """
    Returns the performance of all successfully trained solutions so far. The performances are returned as a dictionary
    mapping the 'solution ID' to the performance score. Use this function to remind yourself of the performance
    of all solutions, so that you can do things such as select the best performing solution for deployment.

    Returns:
        A dictionary mapping solution IDs to their performance scores with structure:
        {
            "solution_id_1": performance_score_1,
            "solution_id_2": performance_score_2,
        }
    """
    from plexe.core.object_registry import ObjectRegistry
    from plexe.core.entities.solution import Solution

    object_registry = ObjectRegistry()
    performances = {}

    for solution_id in object_registry.list_by_type(Solution):
        solution = object_registry.get(Solution, solution_id)
        if solution.performance is not None and solution.performance.value is not None:
            performances[solution_id] = solution.performance.value

    return performances
