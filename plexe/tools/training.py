"""
Tools related to code generation, including solution planning, training code, 
and inference code generation.
"""

import logging
from typing import List, Callable

from smolagents import tool

from plexe.core.object_registry import ObjectRegistry
from plexe.internal.common.provider import Provider
from plexe.internal.models.entities.code import Code
from plexe.internal.models.generation.training import TrainingCodeGenerator

logger = logging.getLogger(__name__)


@tool
def register_best_training_code(best_training_code_id: str) -> str:
    """
    Register the identifier returned by the MLEngineer for the solution with the best performance in the object
    registry. This step is required in order for the training code to be available for future use.

    Args:
        best_training_code_id: 'training_code_id' of the best performing model

    Returns:
        Success message confirming registration
    """
    object_registry = ObjectRegistry()

    try:
        # Register the testing code with a fixed ID
        code_id = "best_performing_training_code"
        code = object_registry.get(Code, best_training_code_id).code
        object_registry.register(Code, code_id, Code(code), overwrite=True, immutable=True)

        logger.debug(f"✅ Registered model training code with ID '{code_id}'")
        return f"Successfully registered model training code with ID '{code_id}' for the best performing model."

    except Exception as e:
        logger.warning(f"⚠️ Error registering training code: {str(e)}")
        raise RuntimeError(f"Failed to register training code: {str(e)}")


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
