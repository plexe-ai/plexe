"""
This module defines agent tools for evaluating the properties and performance of models.
"""

import logging
from typing import Dict, Callable

from smolagents import tool

from plexe.internal.common.provider import Provider
from plexe.internal.models.entities.code import Code
from plexe.internal.models.generation.review import ModelReviewer

logger = logging.getLogger(__name__)


def get_review_finalised_model(llm_to_use: str) -> Callable:
    """Returns a tool function to review finalized models with the model ID pre-filled."""

    @tool
    def review_finalised_model(
        intent: str,
        solution_plan: str,
    ) -> dict:
        """
        Reviews the entire model and extracts metadata. Use this function once you have completed work on the model, and
        you want to 'wrap up' the work by performing a holistic review of what has been built.

        Args:
            intent: The model intent
            solution_plan: The solution plan explanation based on which the model was implemented

        Returns:
            A dictionary containing a summary and review of the model
        """
        from plexe.core.object_registry import ObjectRegistry

        object_registry = ObjectRegistry()

        try:
            input_schema = object_registry.get(dict, "input_schema")
            output_schema = object_registry.get(dict, "output_schema")
        except Exception:
            raise ValueError("Failed to retrieve schemas. Was schema resolution completed?")

        try:
            training_code = object_registry.get(Code, "best_performing_training_code")
        except Exception:
            raise ValueError("Best performing training code not found. Was the best model selected?")

        try:
            inference_code = object_registry.get(Code, "final_inference_code_for_production")
        except Exception:
            raise ValueError("Inference code not found. Was the inference code produced?")

        reviewer = ModelReviewer(Provider(llm_to_use))
        return reviewer.review_model(
            intent, input_schema, output_schema, solution_plan, training_code.code, inference_code.code
        )

    return review_finalised_model


@tool
def get_model_performances() -> Dict[str, float]:
    """
    Returns the performance of all successfully trained models so far. The performances are returned as a dictionary
    mapping the 'model training ID' to the performance score. Use this function to remind yourself of the performance
    of all models, so that you can do things such as select the best performing model for deployment.

    Returns:
        A dictionary mapping model IDs to their performance scores with structure:
        {
            "model_training_id_1": performance_score_1,
            "model_training_id_2": performance_score_2,
        }
    """
    from plexe.core.object_registry import ObjectRegistry

    object_registry = ObjectRegistry()
    performances = {}

    for code_id in object_registry.list_by_type(Code):
        code = object_registry.get(Code, code_id)
        if code.performance is not None:
            performances[code_id] = code.performance

    return performances
