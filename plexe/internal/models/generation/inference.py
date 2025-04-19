# plexe/internal/models/generation/inference.py

"""
This module provides functionality for generating inference code for machine learning models.
"""

import json
from typing import List, Dict, Type

from pydantic import BaseModel

from plexe.config import code_templates, prompt_templates
from plexe.internal.common.provider import Provider
from plexe.internal.common.utils.response import extract_code


class InferenceCodeGenerator:
    def __init__(self, provider: Provider):
        """
        Initializes the InferenceCodeGenerator with an empty context.
        :param provider: the LLM provider to use for querying
        """
        self.provider: Provider = provider
        self.context: List[Dict[str, str]] = []

    def generate_inference_code(
        self, input_schema: Type[BaseModel], output_schema: Type[BaseModel], training_code: str
    ) -> str:
        """
        Generates inference code based on the problem statement, solution plan, and training code.

        :param input_schema: The schema of the input data.
        :param output_schema: The schema of the output data.
        :param training_code: The training code that has already been generated.
        :return: The generated inference code.
        """
        # Stage 1: Generate model loading code
        inference_script = self._generate_model_loading(training_code)

        # Stage 2: Generate preprocessing code
        inference_script = self._generate_preprocessing(inference_script, input_schema, training_code)

        # Stage 3: Generate postprocessing code
        inference_script = self._generate_postprocessing(inference_script, output_schema, training_code)

        # Stage 3: Generate prediction code with context from previous stages
        inference_script = self._generate_prediction(output_schema, training_code, input_schema, inference_script)

        # Combine the stages
        return self._combine_code_stages(inference_script)

    def fix_inference_code(self, inference_code: str, review: str, problems: str) -> str:
        """
        Fixes the inference code based on the review and identified problems.

        :param [str] inference_code: The previously generated inference code.
        :param [str] review: The review of the previous solution.
        :param [str] problems: Specific errors or bugs identified.
        :return str: The fixed inference code.
        """

        class FixResponse(BaseModel):
            plan: str
            code: str

        response: FixResponse = FixResponse(
            **json.loads(
                self.provider.query(
                    system_message=prompt_templates.inference_system(),
                    user_message=prompt_templates.inference_fix(
                        predictor_interface_source=code_templates.predictor_interface,
                        predictor_template=code_templates.predictor_template,
                        inference_code=inference_code,
                        review=review,
                        problems=problems,
                    ),
                    response_format=FixResponse,
                )
            )
        )
        return extract_code(response.code)

    def review_inference_code(
        self,
        inference_code: str,
        input_schema: Type[BaseModel],
        output_schema: Type[BaseModel],
        training_code: str,
        problems: str = None,
    ) -> str:
        """
        Reviews the inference code to identify improvements and fix issues.

        :param [str] inference_code: The previously generated inference code.
        :param [Type[BaseModel]] input_schema: The schema of the input data.
        :param [Type[BaseModel]] output_schema: The schema of the output data.
        :param [str] training_code: The training code that has already been generated.
        :param [str] problems: Specific errors or bugs identified.
        :return: The review of the inference code with suggestions for improvements.
        """
        return self.provider.query(
            system_message=prompt_templates.inference_system(),
            user_message=prompt_templates.inference_review(
                predictor_interface_source=code_templates.predictor_interface,
                predictor_template=code_templates.predictor_template,
                inference_code=inference_code,
                input_schema=input_schema.model_fields,
                output_schema=output_schema.model_fields,
                training_code=training_code,
                problems=problems,
            ),
        )

    def generate_inference_tests(
        self, problem_statement: str, plan: str, training_code: str, inference_code: str
    ) -> str:
        raise NotImplementedError("Generation of the inference tests is not yet implemented.")

    def fix_inference_tests(self, inference_tests: str, inference_code: str, review: str, problems: str) -> str:
        raise NotImplementedError("Fixing of the inference tests is not yet implemented.")

    def review_inference_tests(
        self, inference_tests: str, inference_code: str, problem_statement: str, plan: str
    ) -> str:
        raise NotImplementedError("Review of the inference tests is not yet implemented.")

    def _generate_model_loading(self, training_code: str) -> str:
        """
        Generate code for loading the trained model files.

        :param training_code: The training code to analyze for model saving patterns
        :return: Code snippet for model loading
        """
        return extract_code(
            self.provider.query(
                system_message=prompt_templates.inference_system(),
                user_message=prompt_templates.inference_load(
                    predictor_template=code_templates.predictor_template,
                    training_code=training_code,
                ),
            )
        )

    def _generate_preprocessing(self, inference_code: str, input_schema: Type[BaseModel], training_code: str) -> str:
        """
        Generate code for preprocessing input data before prediction.

        :param inference_code: The previously generated inference code
        :param input_schema: Schema defining the input data format
        :param training_code: Training code to analyze for preprocessing steps
        :return: Code snippet for preprocessing
        """
        return extract_code(
            self.provider.query(
                system_message=prompt_templates.inference_system(),
                user_message=prompt_templates.inference_preprocess(
                    inference_code=inference_code, input_schema=input_schema.model_fields, training_code=training_code
                ),
            )
        )

    def _generate_postprocessing(self, inference_code: str, output_schema: Type[BaseModel], training_code: str) -> str:
        """
        Generate code for postprocessing the model's output after prediction.

        :param inference_code: The previously generated inference code
        :param output_schema: Schema defining the expected output format
        :param training_code: Training code to analyze for postprocessing steps
        :return: Code snippet for postprocessing
        """
        return extract_code(
            self.provider.query(
                system_message=prompt_templates.inference_system(),
                user_message=prompt_templates.inference_postprocess(
                    inference_code=inference_code, output_schema=output_schema.model_fields, training_code=training_code
                ),
            )
        )

    def _generate_prediction(
        self, output_schema: Type[BaseModel], training_code: str, input_schema: Type[BaseModel], inference_code: str
    ) -> str:
        """
        Generate code for making predictions with the loaded model.

        :param [Type[BaseModel]] output_schema: Schema defining the expected output format
        :param [str] training_code: Training code to analyze for prediction patterns
        :param [Type[BaseModel]] input_schema: Schema defining the input data format
        :param [str] inference_code: The previously generated inference code
        :return: Code snippet for prediction
        """
        return extract_code(
            self.provider.query(
                system_message=prompt_templates.inference_system(),
                user_message=prompt_templates.inference_predict(
                    output_schema=output_schema.model_fields,
                    input_schema=input_schema.model_fields,
                    training_code=training_code,
                    inference_code=inference_code,
                ),
            )
        )

    def _combine_code_stages(self, inference_code: str) -> str:
        """
        Combine the separately generated code stages into a complete inference script.

        :param inference_code: The previously generated inference code
        :return: Complete inference script
        """
        return extract_code(
            self.provider.query(
                system_message=prompt_templates.inference_system(),
                user_message=prompt_templates.inference_combine(
                    inference_code=inference_code,
                    predictor_interface_source=code_templates.predictor_interface,
                ),
            )
        )
