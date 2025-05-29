"""
Tools related to code generation, including solution planning, training code, 
and inference code generation.
"""

import logging
from typing import List, Callable

from smolagents import tool

from plexe.core.object_registry import ObjectRegistry
from plexe.internal.common.provider import Provider
from plexe.internal.models.generation.training import TrainingCodeGenerator

logger = logging.getLogger(__name__)


@tool
def register_best_solution(best_solution_id: str) -> str:
    """
    Register the solution with the best performance as the final selected solution in the object
    registry. This step is required in order for the solution to be available for final model building.

    Args:
        best_solution_id: 'solution_id' of the best performing solution

    Returns:
        Success message confirming registration
    """
    from plexe.core.entities.solution import Solution

    object_registry = ObjectRegistry()

    try:
        # Get the best solution
        best_solution = object_registry.get(Solution, best_solution_id)

        # Register the solution with a fixed ID for easy retrieval
        object_registry.register(Solution, "best_performing_solution", best_solution, overwrite=True)

        logger.debug(f"✅ Registered best performing solution with ID '{best_solution_id}'")
        return f"Successfully registered solution with ID '{best_solution_id}' as the best performing solution."

    except Exception as e:
        logger.warning(f"⚠️ Error registering best solution: {str(e)}")
        raise RuntimeError(f"Failed to register best solution: {str(e)}")


def get_training_code_generation_tool(llm_to_use: str) -> Callable:
    """Returns a tool function to generate training code with the model ID pre-filled."""

    @tool
    def generate_training_code(
        task: str, solution_plan: str, train_datasets: List[str], validation_datasets: List[str]
    ) -> str:
        """Generates training code based on the solution plan.

        Args:
            task: The task definition
            solution_plan: The solution plan to implement
            train_datasets: Keys of datasets to use for training
            validation_datasets: Keys of datasets to use for validation

        Returns:
            Generated training code as a string
        """
        train_generator = TrainingCodeGenerator(Provider(llm_to_use))
        return train_generator.generate_training_code(task, solution_plan, train_datasets, validation_datasets)

    return generate_training_code


def get_training_code_fixing_tool(llm_to_use: str) -> Callable:
    """Returns a tool function to fix training code with the model ID pre-filled."""

    @tool
    def fix_training_code(
        training_code: str,
        solution_plan: str,
        review: str,
        train_datasets: List[str],
        validation_datasets: List[str],
        issue: str,
    ) -> str:
        """
        Fixes issues in the training code based on a review.

        Args:
            training_code: The training code to fix
            solution_plan: The solution plan being implemented
            review: Review comments about the code and its issues, ideally a summary analysis of the issue
            train_datasets: Keys of datasets to use for training
            validation_datasets: Keys of datasets to use for validation
            issue: Description of the issue to address

        Returns:
            Fixed training code as a string
        """
        train_generator = TrainingCodeGenerator(Provider(llm_to_use))
        return train_generator.fix_training_code(
            training_code, solution_plan, review, train_datasets, validation_datasets, issue
        )

    return fix_training_code
