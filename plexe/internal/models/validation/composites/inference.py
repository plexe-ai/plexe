"""
This module defines a composite validator for validating the correctness of prediction code.

Classes:
    - InferenceCodeValidator: A validator class that validates the correctness of prediction code.
"""

import pandas as pd
from typing import Type

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
        input_sample: pd.DataFrame,
    ):
        """
        Initialize the PredictionValidator with the name 'prediction'.
        """
        super().__init__(
            "prediction",
            [
                SyntaxValidator(),
                PredictorValidator(input_schema, output_schema, input_sample),
            ],
        )
