# plexe/internal/models/validation/validator.py

"""
This module defines the `Validator` abstract base class and the `ValidationResult` data class.

The `Validator` class provides a framework for implementing various code validators, while the
`ValidationResult` class encapsulates the results of a validation, including whether it passed,
any messages, and exceptions raised during validation.
"""

import abc
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """
    Represents the result of a validation.

    Attributes:
        name (str): The name of the validation.
        passed (bool): Whether the validation passed or not.
        message (str, optional): A message providing details about the validation result.
        exception (Exception, optional): An exception that was raised during validation, if any.
        error_stage (str, optional): The stage at which validation failed (e.g., "syntax", "loading", "prediction").
        error_type (str, optional): The type of error that occurred (e.g., "SyntaxError", "TypeError").
        error_details (str, optional): Detailed information about the error.
    """

    name: str
    passed: bool
    message: str = None
    exception: Exception | None = None
    error_stage: str = None
    error_type: str = None
    error_details: str = None


class Validator(abc.ABC):
    """
    Abstract base class for validators.

    Attributes:
        name (str): The name of the validator.
    """

    @abc.abstractmethod
    def __init__(self, name: str):
        """
        Initializes the validator with a name.

        :param [str] name: The name of the validator.
        """
        self.name = name

    @abc.abstractmethod
    def validate(self, code: str, **kwargs) -> ValidationResult:
        """
        Validates the given code.

        :param [str] code: The code to validate.
        :return: [ValidationResult] The result of the validation.
        """
        pass
