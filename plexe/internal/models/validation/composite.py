# plexe/internal/models/validation/composite.py

"""
This module defines the `CompositeValidator` class, which chains multiple validators together in a workflow.

Classes:
    - CompositeValidator: A validator that chains multiple validators together in a workflow.
"""

from plexe.internal.models.validation.validator import Validator, ValidationResult


class CompositeValidator(Validator):
    """
    A validator that chains multiple validators together in a workflow.

    Attributes:
        validators (list[Validator]): The validators to run in the workflow.
    """

    def __init__(self, name: str, validators: list[Validator]):
        """
        Initializes the validator pipeline with a name and a list of validators.

        :param [str] name: The name of the validator pipeline.
        :param [list[Validator]] validators: The validators to run in the pipeline.
        """
        super().__init__(name)
        self.validators = validators

    def validate(self, code: str, **kwargs) -> ValidationResult:
        """
        Validates the given code by running it through each validator in the pipeline.

        :param [str] code: The code to validate.
        :return: [ValidationResult] The result of the validation.
        """
        for validator in self.validators:
            result = validator.validate(code, **kwargs)
            if not result.passed:
                return result

        return ValidationResult(self.name, True, "All validators passed.")
