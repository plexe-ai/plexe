# plexe/errors.py
"""
This module defines custom exception classes for the plexe library.

The exceptions are categorized into three main groups:

1. **Specification-related errors**:
   - Handle issues with natural language specifications and schemas.

2. **Instruction-related errors**:
   - Capture problems with invalid or improperly defined model-building instructions.

3. **Constraint-related errors**:
   - Manage violations of user-defined constraints on models.

Each error class inherits from the base `PlexeError` to allow library-wide exception handling.
"""


class PlexeError(Exception):
    """
    Base class for all errors in the plexe library.
    """

    pass


# Specification-related errors
class SpecificationError(PlexeError):
    """
    Base class for errors related to model specification.
    """

    pass


class InsufficientSpecificationError(SpecificationError):
    """
    Raised when the natural language specification is insufficiently detailed.
    """

    pass


class AmbiguousSpecificationError(SpecificationError):
    """
    Raised when the natural language specification is ambiguous or contradictory.
    """

    pass


class InvalidSchemaError(SpecificationError):
    """
    Raised when the input or output schema is invalid.
    """

    pass


# Instruction-related errors
class InstructionError(PlexeError):
    """
    Base class for errors related to instructions provided for model building.
    """

    pass


# todo: add more specific instruction-related errors once we have a better idea of what instructions are


# Constraint-related errors
class ConstraintError(PlexeError):
    """
    Base class for errors related to constraints.
    """

    pass


# todo: add more specific constraint-related errors once we have a better idea of how constraints are used


# Runtime-related errors
class PlexeRuntimeError(PlexeError, RuntimeError):
    """
    Base class for runtime errors during model execution or training.
    """

    pass


class CodeExecutionError(PlexeRuntimeError):
    """
    Raised when code execution fails for reasons other than timeout.
    """

    pass
