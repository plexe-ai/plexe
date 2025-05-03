"""
Tools related to code generation, including solution planning, training code, 
and inference code generation.
"""

import logging
from typing import List, Callable

from smolagents import tool

from plexe.internal.common.provider import Provider
from plexe.internal.models.generation.training import TrainingCodeGenerator

logger = logging.getLogger(__name__)


def get_generate_training_code(llm_to_use: str) -> Callable:
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


def get_fix_training_code(llm_to_use: str) -> Callable:
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
