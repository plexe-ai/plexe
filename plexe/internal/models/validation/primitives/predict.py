"""
This module defines the `PredictorValidator` class, which validates that a predictor behaves as expected.

Classes:
    - PredictorValidator: A validator class that checks the behavior of a predictor.
"""

import types
import warnings
from typing import Type
import pandas as pd

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
        sample: pd.DataFrame,
    ) -> None:
        """
        Initialize the PredictorValidator with the name 'predictor'.

        :param input_schema: The input schema of the predictor.
        :param output_schema: The output schema of the predictor.
        :param sample: The sample input data to test the predictor.
        """
        super().__init__("predictor")
        self.input_schema: Type[BaseModel] = input_schema
        self.output_schema: Type[BaseModel] = output_schema
        self.input_sample: pd.DataFrame = sample

    def validate(self, code: str, model_artifacts=None) -> ValidationResult:
        """
        Validates that the given code for a predictor behaves as expected.
        :param code: prediction code to be validated
        :param model_artifacts: model artifacts to be used for validation
        :return: True if valid, False otherwise
        """
        try:
            predictor_module: types.ModuleType = self._load_module(code)
            predictor_class = getattr(predictor_module, "PredictorImplementation")
            predictor = predictor_class(model_artifacts)
            self._is_subclass(predictor_class)
            self._returns_output_when_called(predictor)

            return ValidationResult(self.name, True, "Prediction code is valid.")

        except Exception as e:
            return ValidationResult(
                self.name,
                False,
                message=f"Prediction code is not valid: {str(e)}.",
                exception=e,
            )

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
        Tests the `predict` function by calling it with sample inputs.
        """
        total_tests = len(self.input_sample)
        issues = []

        for i, sample in self.input_sample.iterrows():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    predictor.predict(sample.to_dict())
            except Exception as e:
                issues.append({"error": str(e), "sample": sample, "index": i})

        if len(issues) > 0:
            raise RuntimeError(f"{len(issues)}/{total_tests} calls to 'predict' failed. Issues: {issues}")
