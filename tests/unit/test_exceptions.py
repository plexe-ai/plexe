"""
Tests for the custom exception classes defined in the plexe.errors module.

These tests primarily validate the inheritance relationships between exceptions and ensure
that the exceptions can be raised and caught correctly. While these classes currently lack
custom fields or methods, these tests help prevent regressions in the definitions of the errors.
"""

import pytest

from plexe.exceptions import (
    PlexeError,
    SpecificationError,
    InsufficientSpecificationError,
    AmbiguousSpecificationError,
    InvalidSchemaError,
    InstructionError,
    ConstraintError,
)


def test_base_error():
    with pytest.raises(PlexeError) as exc_info:
        raise PlexeError("Base error")
    assert str(exc_info.value) == "Base error"


def test_specification_errors():
    with pytest.raises(SpecificationError) as exc_info:
        raise SpecificationError("Specification error")
    assert str(exc_info.value) == "Specification error"

    with pytest.raises(InsufficientSpecificationError) as exc_info:
        raise InsufficientSpecificationError("Insufficient detail")
    assert str(exc_info.value) == "Insufficient detail"

    with pytest.raises(AmbiguousSpecificationError) as exc_info:
        raise AmbiguousSpecificationError("Ambiguous specification")
    assert str(exc_info.value) == "Ambiguous specification"

    with pytest.raises(InvalidSchemaError) as exc_info:
        raise InvalidSchemaError("Invalid schema")
    assert str(exc_info.value) == "Invalid schema"


def test_instruction_errors():
    with pytest.raises(InstructionError) as exc_info:
        raise InstructionError("Instruction error")
    assert str(exc_info.value) == "Instruction error"


def test_constraint_errors():
    with pytest.raises(ConstraintError) as exc_info:
        raise ConstraintError("Constraint error")
    assert str(exc_info.value) == "Constraint error"


def test_inheritance_relationships():
    # Check SpecificationError hierarchy
    assert issubclass(SpecificationError, PlexeError)
    assert issubclass(InsufficientSpecificationError, SpecificationError)
    assert issubclass(AmbiguousSpecificationError, SpecificationError)
    assert issubclass(InvalidSchemaError, SpecificationError)

    # Check InstructionError hierarchy
    assert issubclass(InstructionError, PlexeError)

    # Check ConstraintError hierarchy
    assert issubclass(ConstraintError, PlexeError)
