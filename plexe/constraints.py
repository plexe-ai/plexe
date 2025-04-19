"""
constraint.py

This module defines the `Constraint` class, which represents a rule or condition that a function's
input/output intent must satisfy. Constraints are used to specify that a function's intent
adhere to certain rules, enabling a concise mechanism for providing context about business rule and
validating function outputs.

Each `Constraint` consists of:
- A `condition`: A callable that takes an input/output pair and evaluates whether the constraint is satisfied.
- An optional `description`: A human-readable description of the constraint.

The `Constraint` class provides composability through logical operators:
- `&` (AND): Combines two constraints, both of which must be satisfied.
- `|` (OR): Combines two constraints, either of which must be satisfied.
- `~` (NOT): Negates a constraint, requiring it to be unsatisfied.

Usage:
    from constraint import Constraint

    # Define a simple constraint
    non_negative = Constraint(
        condition=lambda inputs, outputs: outputs["score"] >= 0,
        description="The score must be non-negative"
    )

    # Combine constraints
    has_winner = Constraint(
        condition=lambda inputs, outputs: "winner" in outputs,
        description="Output must contain a winner"
    )

    valid_constraints = non_negative & has_winner

    # Evaluate the constraints
    input_data = {"player1": "Alice", "player2": "Bob"}
    output_data = {"score": 10, "winner": "Alice"}
    result = valid_constraints.evaluate(input_data, output_data)
    print(result)  # True
"""

import inspect
from typing import Any, Callable, Optional


# todo: something to think about is how to represent constraints in cases where the model is not a
# deterministic function, but rather represents a probability distribution P(Y|X), or even a distribution
# P(X^Y) for a generative model. In these cases, a strict boolean constraint on input/output pairs may not
# make sense.
class Constraint:
    """
    Represents a constraint on a function's input/output intent.

    A `Constraint` is defined by a `condition`, a callable that evaluates whether the constraint
    is satisfied for a given input/output pair, and an optional `description` that provides
    a human-readable explanation of the constraint.

    Constraints are composable, allowing users to combine them using logical operators:
    - `&` (AND): Combines two constraints such that both must be satisfied.
    - `|` (OR): Combines two constraints such that either must be satisfied.
    - `~` (NOT): Negates a constraint, requiring it to be unsatisfied.

    Attributes:
        condition (Callable[[Any, Any], bool]): A function that takes (inputs, outputs) and evaluates
            whether the constraint is satisfied.
        description (str): A human-readable description of the constraint.

    Methods:
        evaluate(inputs, outputs): Evaluates the constraint on a given inputs/outputs pair.

    Example:
        non_negative = Constraint(
            condition=lambda inputs, outputs: outputs["score"] >= 0,
            description="The score must be non-negative"
        )

        input_data = {"player1": "Alice", "player2": "Bob"}
        output_data = {"score": 10, "winner": "Alice"}

        print(non_negative.evaluate(input_data, output_data))  # True
    """

    def __init__(self, condition: Callable[[Any, Any], bool], description: Optional[str] = None):
        """
        Initialize a constraint with an optional description and a condition.

        :param Callable[[Any, Any], bool] condition: A function that takes (inputs, outputs) and returns
            True if the constraint is satisfied.
        :param Optional[str] description: A human-readable description of the constraint. Defaults to None.
        :raises TypeError: If the condition is not callable.
        :raises ValueError: If the condition does not accept exactly two arguments.
        """
        # Validate that condition is a callable
        if not callable(condition):
            raise TypeError(
                "Condition must be a callable that takes two arguments (inputs, outputs) and returns a boolean."
            )

        # Optionally, check the callable signature (if you want stricter validation)
        signature = inspect.signature(condition)
        if len(signature.parameters) != 2:
            raise ValueError("Condition must be a callable that accepts exactly two arguments: (inputs, outputs).")

        self.condition = condition
        self.description = description or "Unnamed constraint"

    def evaluate(self, inputs: Any, outputs: Any) -> bool:
        """
        Evaluate the constraint on a given inputs/outputs pair.

        :param Any inputs: The inputs to the model.
        :param Any outputs: The outputs from the model.
        :return: True if the constraint is satisfied, False otherwise.
        :raises RuntimeError: If an error occurs during constraint evaluation.
        """
        try:
            return self.condition(inputs, outputs)
        except Exception as e:
            raise RuntimeError(f"Constraint evaluation failed: {self.description}") from e

    def __str__(self) -> str:
        return f"Constraint(description={self.description})"

    def __and__(self, other: "Constraint") -> "Constraint":
        """
        Combine this constraint with another using logical AND.

        :param Constraint other: Another constraint to combine with.
        :return: A new constraint that is satisfied if both constraints are satisfied.
        """
        return Constraint(
            condition=lambda inputs, outputs: self.evaluate(inputs, outputs) and other.evaluate(inputs, outputs),
            description=f"({self.description}) AND ({other.description})",
        )

    def __or__(self, other: "Constraint") -> "Constraint":
        """
        Combine this constraint with another using logical OR.

        :param Constraint other: Another constraint to combine with.
        :return: A new constraint that is satisfied if either constraint is satisfied.
        """
        return Constraint(
            condition=lambda inputs, outputs: self.evaluate(inputs, outputs) or other.evaluate(inputs, outputs),
            description=f"({self.description}) OR ({other.description})",
        )

    def __invert__(self) -> "Constraint":
        """
        Negate this constraint (logical NOT).

        :return: A new constraint that is satisfied if this constraint is not satisfied.
        """
        return Constraint(
            condition=lambda inputs, outputs: not self.evaluate(inputs, outputs),
            description=f"NOT ({self.description})",
        )
