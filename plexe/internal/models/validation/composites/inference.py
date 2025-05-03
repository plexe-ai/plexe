"""
This module defines a composite validator for validating the correctness of prediction code.

Classes:
    - InferenceCodeValidator: A validator class that validates the correctness of prediction code.
"""

from typing import Type, List, Dict, Any

from pydantic import BaseModel

from plexe.internal.models.validation.composite import CompositeValidator
from plexe.internal.models.validation.primitives.predict import PredictorValidator
from plexe.internal.models.validation.primitives.syntax import SyntaxValidator


class InferenceCodeValidator(CompositeValidator):
    """
    A validator class that validates the correctness of prediction code.
    """

    def __init__(
        self,
        input_schema: Type[BaseModel],
        output_schema: Type[BaseModel],
        input_sample: List[Dict[str, Any]],
    ):
        """
        Initialize the InferenceCodeValidator with the name 'prediction'.

        Args:
            input_schema: The input schema for the model
            output_schema: The output schema for the model
            input_sample: List of sample input dictionaries to test the predictor
        """
        super().__init__(
            "prediction",
            [
                SyntaxValidator(),
                PredictorValidator(input_schema, output_schema, input_sample),
            ],
        )
