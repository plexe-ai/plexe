"""
This module provides utility functions for defining and managing tools for AI agents.
"""

import inspect
from functools import wraps


class AgentInvocationError(Exception):
    """Raised when an agent calls an @agentinspectable function with invalid arguments."""

    def __init__(self, func_name: str, help_text: str):
        message = f"Incorrect arguments for function '{func_name}'.\n\n{help_text}"
        super().__init__(message)
        self.func_name = func_name
        self.help_text = help_text


def agentinspectable(func):
    """
    Decorator for functions intended to be made available for calling to AI agents using mechanisms
    other than standard 'tool' instantiation.

    If the function is called with the wrong arguments (e.g. missing or extra params), instead of raising
    a TypeError it will return its own docstring and a usage string. This allows an agent to "inspect" the
    function by trial-and-error and learn how to call it correctly.

    This decorator was originally conceived as a mechanism to 'inject' arbitrary functions into smolagents
    CodeAgent agents by passing references to them directly into the agent's code execution environment, while
    at the same time benefiting from function docstrings in the same way as standard tools with the @tool
    decorator. When a function is passed to a CodeAgent this way, the agent can call it directly, but does
    not have access to the function signature. This decorator allows the agent to discover the correct
    signature by trial-and-error.

    Example:
        @agentinspectable
        def greet(name: str, times: int = 1):
            "greet(name: str, times: int = 1) -> prints greetings"
            for _ in range(times):
                print(f"Hello, {name}!")

        greet("Marcello", 2)  # works
        greet()               # returns docstring + usage
    """
    sig = inspect.signature(func)
    usage = f"{func.__name__}{sig}"
    doc = inspect.getdoc(func) or ""
    help_text = (doc + ("\n\n" if doc else "") + f"Usage: {usage}").strip()

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            sig.bind(*args, **kwargs)
        except TypeError:
            raise AgentInvocationError(func.__name__, help_text)
        return func(*args, **kwargs)

    wrapper.__agentinspectable__ = True
    wrapper.__usage__ = usage
    return wrapper
