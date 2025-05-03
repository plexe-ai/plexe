"""
This module defines the `PredictorValidator` class, which validates that a predictor behaves as expected.

Classes:
    - PredictorValidator: A validator class that checks the behavior of a predictor.
"""

import types
import warnings
from typing import Type, List, Dict, Any

from pydantic import BaseModel

from plexe.internal.models.validation.validator import Validator, ValidationResult
from plexe.internal.models.interfaces.predictor import Predictor


class PredictorValidator(Validator):
    """
    A validator class that checks that a predictor behaves as expected.
    """

    def __init__(
        self,
        input_schema: Type[BaseModel],
        output_schema: Type[BaseModel],
        sample: List[Dict[str, Any]],
    ) -> None:
        """
        Initialize the PredictorValidator with the name 'predictor'.

        :param input_schema: The input schema of the predictor.
        :param output_schema: The output schema of the predictor.
        :param sample: List of sample input dictionaries to test the predictor.
        """
        super().__init__("predictor")
        self.input_schema: Type[BaseModel] = input_schema
        self.output_schema: Type[BaseModel] = output_schema
        self.input_sample: List[Dict[str, Any]] = sample

    def validate(self, code: str, model_artifacts=None) -> ValidationResult:
        """
        Validates that the given code for a predictor behaves as expected.
        :param code: prediction code to be validated
        :param model_artifacts: model artifacts to be used for validation
        :return: True if valid, False otherwise
        """

        def validation_error(stage, e):
            """Helper to create validation error results"""
            return ValidationResult(
                self.name,
                False,
                message=f"Failed at {stage} stage: {str(e)}",
                exception=e,
                error_stage=stage,
                error_type=type(e).__name__,
                error_details=str(e),
            )

        # Stage 1: Load module
        try:
            predictor_module = self._load_module(code)
        except Exception as e:
            return validation_error("loading", e)

        # Stage 2: Check class definition
        try:
            predictor_class = getattr(predictor_module, "PredictorImplementation")
            self._is_subclass(predictor_class)
        except Exception as e:
            return validation_error("class_definition", e)

        # Stage 3: Initialize predictor
        try:
            predictor = predictor_class(model_artifacts)
        except Exception as e:
            return validation_error("initialization", e)

        # Stage 4: Test prediction
        try:
            self._returns_output_when_called(predictor)
        except Exception as e:
            return validation_error("prediction", e)

        # All validation steps passed
        return ValidationResult(self.name, True, "Prediction code is valid.")

    @staticmethod
    def _load_module(code: str) -> types.ModuleType:
        """
        Compiles and loads the predictor module from the given code.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            module = types.ModuleType("test_predictor")
            try:
                exec(code, module.__dict__)
            except Exception as e:
                raise RuntimeError(f"Failed to load predictor: {str(e)}")
        return module

    @staticmethod
    def _is_subclass(predictor) -> None:
        if not issubclass(predictor, Predictor):
            raise TypeError("The predictor class is not a subclass of Predictor.")

    def _returns_output_when_called(self, predictor) -> None:
        """
        Tests the `predict` function by calling it with sample inputs and validates outputs.
        """
        total_tests = len(self.input_sample)
        issues = []

        for i, sample in enumerate(self.input_sample):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Test prediction execution
                    output = predictor.predict(sample)

                    # Validate output against schema
                    try:
                        self.output_schema.model_validate(output)
                    except Exception as schema_err:
                        # Include truncated sample and output for context
                        truncated_output = str(output)[:100] + "..." if len(str(output)) > 100 else output
                        issues.append(
                            {
                                "error": f"Output schema validation error: {str(schema_err)}",
                                "output": truncated_output,
                                "index": i,
                            }
                        )
            except Exception as e:
                # Include truncated sample for context
                sample_str = str(sample)
                truncated_sample = sample_str[:100] + "..." if len(sample_str) > 100 else sample_str
                issues.append({"error": str(e), "sample": truncated_sample, "index": i})

        if len(issues) > 0:
            raise RuntimeError(f"{len(issues)}/{total_tests} calls to 'predict' failed. Issues: {issues}")
