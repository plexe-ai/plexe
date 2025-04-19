"""
Tests for the Constraint class in plexe.constraints.

This module verifies:
1. Validation of conditions during initialization.
2. Logical operations (AND, OR, NOT) between constraints.
3. Error handling during condition evaluation.
"""

import pytest
from plexe.constraints import Constraint


def test_valid_condition():
    constraint = Constraint(
        condition=lambda inputs, outputs: outputs.get("value", 0) > 0,
        description="Outputs must have a value greater than 0",
    )
    assert constraint.evaluate({}, {"value": 1}) is True
    assert constraint.evaluate({}, {"value": 0}) is False


def test_invalid_condition_type():
    with pytest.raises(TypeError):
        Constraint(condition="not_callable")


def test_invalid_condition_signature():
    with pytest.raises(ValueError):
        Constraint(condition=lambda x: x > 0)
    with pytest.raises(ValueError):
        Constraint(condition=lambda x, y, z: x > 0)


def test_and_operator():
    constraint1 = Constraint(
        condition=lambda inputs, outputs: outputs.get("value", 0) > 0,
        description="Outputs must have a value greater than 0",
    )
    constraint2 = Constraint(
        condition=lambda inputs, outputs: outputs.get("status") == "success",
        description="Status must be 'success'",
    )
    combined = constraint1 & constraint2
    assert combined.evaluate({}, {"value": 1, "status": "success"}) is True
    assert combined.evaluate({}, {"value": 0, "status": "success"}) is False
    assert combined.evaluate({}, {"value": 1, "status": "failure"}) is False


def test_or_operator():
    constraint1 = Constraint(
        condition=lambda inputs, outputs: outputs.get("value", 0) > 0,
        description="Outputs must have a value greater than 0",
    )
    constraint2 = Constraint(
        condition=lambda inputs, outputs: outputs.get("status") == "success",
        description="Status must be 'success'",
    )
    combined = constraint1 | constraint2
    assert combined.evaluate({}, {"value": 1, "status": "failure"}) is True
    assert combined.evaluate({}, {"value": 0, "status": "success"}) is True
    assert combined.evaluate({}, {"value": 0, "status": "failure"}) is False


def test_not_operator():
    constraint = Constraint(
        condition=lambda inputs, outputs: outputs.get("value", 0) > 0,
        description="Outputs must have a value greater than 0",
    )
    negated = ~constraint
    assert negated.evaluate({}, {"value": 1}) is False
    assert negated.evaluate({}, {"value": 0}) is True


def test_combined_operators():
    constraint1 = Constraint(
        condition=lambda inputs, outputs: outputs.get("value", 0) > 0,
        description="Outputs must have a value greater than 0",
    )
    constraint2 = Constraint(
        condition=lambda inputs, outputs: outputs.get("status") == "success",
        description="Status must be 'success'",
    )
    combined = ~(constraint1 & constraint2)
    assert combined.evaluate({}, {"value": 1, "status": "success"}) is False
    assert combined.evaluate({}, {"value": 0, "status": "success"}) is True
    assert combined.evaluate({}, {"value": 1, "status": "failure"}) is True


def test_runtime_error_handling():
    constraint = Constraint(
        condition=lambda inputs, outputs: outputs["undefined_key"] > 0,
        description="Should raise a KeyError",
    )
    with pytest.raises(RuntimeError):
        constraint.evaluate({}, {})
