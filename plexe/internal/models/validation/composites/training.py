# internal/models/validation/pipelines.py

"""
This module defines a composite validator for validating the correctness of training code.

Classes:
    - TrainingCodeValidator: A validator class that validates the correctness of training code.
"""

from plexe.internal.models.validation.primitives.syntax import SyntaxValidator
from plexe.internal.models.validation.composite import CompositeValidator


class TrainingCodeValidator(CompositeValidator):
    """
    A validator class that validates the correctness of training code.
    """

    def __init__(self):
        """
        Initialize the TrainingValidator with the name 'training'.
        """
        super().__init__("training", [SyntaxValidator()])
