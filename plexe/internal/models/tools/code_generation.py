"""
Tools related to code generation, including solution planning, training code, 
and inference code generation.
"""

import logging
from typing import List, Dict, Callable

from smolagents import tool

from plexe.internal.common.provider import Provider
from plexe.internal.models.entities.code import Code
from plexe.internal.models.generation.inference import InferenceCodeGenerator
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


def get_generate_inference_code(llm_to_use: str) -> Callable:
    """Returns a tool function to generate inference code with the model ID pre-filled."""

    @tool
    def generate_inference_code(
        input_schema: Dict[str, str], output_schema: Dict[str, str], training_code_id: str
    ) -> str:
        """
        Generates inference code based on the training code. The schemas must be provided as a flat dictionary
        mapping field names to strings representing their types (e.g., "int", "str").

        Args:
            input_schema: The input schema for the model, for example {"feat_1": "int", "feat_2": "str"}
            output_schema: The output schema for the model, for example {"output": "float"}
            training_code_id: The identifier for the training code that was used to train the model

        Returns:
            Generated inference code as a string
        """
        from plexe.internal.common.utils.pydantic_utils import map_to_basemodel
        from plexe.internal.common.registries.objects import ObjectRegistry

        try:
            training_code = ObjectRegistry().get(Code, training_code_id).code
        except Exception:
            raise ValueError(
                f"Training code with ID {training_code_id} not found. Did your manager provide the right ID?"
            )

        try:
            # Convert dict schemas to Type[BaseModel]
            input_model = map_to_basemodel("InputSchema", input_schema)
            output_model = map_to_basemodel("OutputSchema", output_schema)

            infer_generator = InferenceCodeGenerator(Provider(llm_to_use))
            return infer_generator.generate_inference_code(input_model, output_model, training_code)
        except Exception as e:
            raise ValueError(f"Failed to generate inference code: {str(e)}") from e

    return generate_inference_code


def get_fix_inference_code(llm_to_use: str) -> Callable:
    """Returns a tool function to fix inference code with the model ID pre-filled."""

    @tool
    def fix_inference_code(inference_code: str, review: str, problems: str) -> str:
        """
        Fixes issues in the inference code based on the review.

        Args:
            inference_code: The inference code to fix
            review: Review comments about the code
            problems: Description of the problems to address

        Returns:
            Fixed inference code as a string
        """
        infer_generator = InferenceCodeGenerator(Provider(llm_to_use))
        return infer_generator.fix_inference_code(inference_code, review, problems)

    return fix_inference_code
