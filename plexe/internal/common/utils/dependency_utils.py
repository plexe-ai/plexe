"""
Utilities for handling optional dependencies.
"""

import functools
import logging
from typing import Callable, TypeVar, cast, Any

from plexe.config import is_package_available

logger = logging.getLogger(__name__)

# Type variable for the callable return type
T = TypeVar("T")


def requires_package(package_name: str, error_message: str = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that checks if a required package is installed before executing a function.
    If the package is not available, logs a warning and returns None.

    Example:
    @requires_package('torch')
    def train_neural_network(data):
        # This function will only run if torch is installed
        import torch
        # ...training code...

    :param package_name: Name of the package that needs to be available
    :param error_message: Custom error message to display if the package is not available
    :return: Decorator function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            if is_package_available(package_name):
                return func(*args, **kwargs)
            else:
                default_message = (
                    f"The '{package_name}' package is required for this functionality but is not installed. "
                    f"Install it with 'pip install plexe[all]' or 'pip install {package_name}'."
                )
                message = error_message or default_message
                logger.warning(message)
                return cast(T, None)

        return wrapper

    return decorator
