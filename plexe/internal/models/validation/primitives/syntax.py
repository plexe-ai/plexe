# plexe/internal/models/validation/syntax.py

"""
This module defines the SyntaxValidator class, which is responsible for validating the syntax
of Python code using the AST module.

Classes:
    - SyntaxValidator: A validator class that checks the syntax of Python code.
"""

import ast

from plexe.internal.models.validation.validator import Validator, ValidationResult


class SyntaxValidator(Validator):
    """
    A validator class that checks the syntax of Python code using the AST module.
    """

    def __init__(self):
        """
        Initialize the SyntaxValidator with the name 'syntax'.
        """
        super().__init__("syntax")

    def validate(self, code: str, **kwargs) -> ValidationResult:
        """
        Validate Python code using AST.

        :param code: Python code to validate.
        :return: Validation result indicating syntax validity.
        """
        try:
            ast.parse(code)
            return ValidationResult(self.name, passed=True, message="Syntax is valid.")
        except SyntaxError as e:
            return ValidationResult(
                self.name,
                False,
                message=f"Syntax is not valid: {e.msg} at line {e.lineno}, column {e.offset}.",
                exception=e,
                error_stage="syntax",
                error_type="SyntaxError",
                error_details=f"{e.msg} at line {e.lineno}, column {e.offset}",
            )
