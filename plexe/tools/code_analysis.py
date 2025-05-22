"""
Tools for analyzing and inspecting code.
"""

import logging
from smolagents import tool

from plexe.core.object_registry import ObjectRegistry
from plexe.internal.models.entities.code import Code

logger = logging.getLogger(__name__)


@tool
def read_training_code(training_code_id: str) -> str:
    """
    Retrieves the training code from the registry for analysis. Use this tool to understand the
    code that was used to train the ML model.

    Args:
        training_code_id: The identifier for the training code to retrieve

    Returns:
        The full training code as a string
    """
    try:
        return ObjectRegistry().get(Code, training_code_id).code
    except Exception as e:
        raise ValueError(f"Failed to retrieve training code with ID {training_code_id}: {str(e)}")
