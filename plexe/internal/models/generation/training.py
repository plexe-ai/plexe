"""
This module provides functions and classes for generating, fixing, and reviewing machine learning model training code.

Functions:
    generate_training_code: Generates machine learning model training code based on a problem statement and solution plan.
    generate_training_tests: Generates tests for the machine learning model training code.
    fix_training_code: Fixes the machine learning model training code based on review and identified problems.
    fix_training_tests: Fixes the tests for the machine learning model training code based on review and identified problems.
    review_training_code: Reviews the machine learning model training code to identify improvements and fix issues.
    review_training_tests: Reviews the tests for the machine learning model training code to identify improvements and fix issues.

Classes:
    TrainingCodeGenerator: A class to generate, fix, and review machine learning model training code.
"""

import json
import logging
from typing import List, Dict
from pathlib import Path

from pydantic import BaseModel

from plexe.config import config, prompt_templates
from plexe.internal.common.provider import Provider
from plexe.internal.common.utils.response import extract_code

logger = logging.getLogger(__name__)


class TrainingCodeGenerator:
    """
    A class to generate, fix, and review machine learning model training code.
    """

    def __init__(self, provider: Provider):
        """
        Initializes the TrainingCodeGenerator with an empty history.

        :param Provider provider: The provider to use for querying.
        """
        self.provider = provider
        self.history: List[Dict[str, str]] = []

    def generate_training_code(
        self,
        problem_statement: str,
        plan: str,
        train_dataset_names: list[str],
        validation_dataset_names: list[str] = None,
    ) -> str:
        """
        Generates machine learning model training code based on the given problem statement and solution plan.

        :param [str] problem_statement: The description of the problem to be solved.
        :param [str] plan: The proposed solution plan.
        :param [str] train_dataset_names: The names of the datasets to use for training.
        :param [str] validation_dataset_names: The names of the datasets to use for validation.
        :return str: The generated training code.
        """
        validation_dataset_names = validation_dataset_names or []

        return extract_code(
            self.provider.query(
                system_message=prompt_templates.training_system(),
                user_message=prompt_templates.training_generate(
                    problem_statement=problem_statement,
                    plan=plan,
                    history=self.history,
                    allowed_packages=config.code_generation.allowed_packages,
                    training_data_files=[Path(f"{file}.parquet").as_posix() for file in train_dataset_names],
                    validation_data_files=[Path(f"{file}.parquet").as_posix() for file in validation_dataset_names],
                ),
            )
        )

    def fix_training_code(
        self,
        training_code: str,
        plan: str,
        review: str,
        train_dataset_names: list[str],
        validation_dataset_names: list[str] = None,
        problems: str = None,
    ) -> str:
        """
        Fixes the machine learning model training code based on the review and identified problems.

        :param [str] training_code: The previously generated training code.
        :param [str] plan: The proposed solution plan.
        :param [str] review: The review of the previous solution.
        :param [str] train_dataset_names: The names of the datasets to use for training.
        :param [str] validation_dataset_names: The names of the datasets to use for validation.
        :param [str] problems: Specific errors or bugs identified.
        :return str: The fixed training code.
        """

        class FixResponse(BaseModel):
            plan: str
            code: str

        response: FixResponse = FixResponse(
            **json.loads(
                self.provider.query(
                    system_message=prompt_templates.training_system(),
                    user_message=prompt_templates.training_fix(
                        plan=plan,
                        training_code=training_code,
                        review=review,
                        problems=problems,
                        training_data_files=[Path(f"{file}.parquet").as_posix() for file in train_dataset_names],
                        validation_data_files=[Path(f"{file}.parquet").as_posix() for file in validation_dataset_names],
                        allowed_packages=config.code_generation.allowed_packages,
                    ),
                    response_format=FixResponse,
                )
            )
        )
        return extract_code(response.code)

    def review_training_code(self, training_code: str, problem_statement: str, plan: str, problems: str = None) -> str:
        """
        Reviews the machine learning model training code to identify improvements and fix issues.

        :param [str] training_code: The previously generated training code.
        :param [str] problem_statement: The description of the problem to be solved.
        :param [str] plan: The proposed solution plan.
        :param [str] problems: Specific errors or bugs identified.
        :return str: The review of the training code with suggestions for improvements.
        """
        return self.provider.query(
            system_message=prompt_templates.training_system(),
            user_message=prompt_templates.training_review(
                problem_statement=problem_statement,
                plan=plan,
                training_code=training_code,
                problems=problems,
                allowed_packages=config.code_generation.allowed_packages,
            ),
        )

    def generate_training_tests(self, problem_statement: str, plan: str, training_code: str) -> str:
        raise NotImplementedError("Generation of the training tests is not yet implemented.")

    def fix_training_tests(self, training_tests: str, training_code: str, review: str, problems: str = None) -> str:
        raise NotImplementedError("Fixing of the training tests is not yet implemented.")

    def review_training_tests(self, training_tests: str, training_code: str, problem_statement: str, plan: str) -> str:
        raise NotImplementedError("Review of the training tests is not yet implemented.")
