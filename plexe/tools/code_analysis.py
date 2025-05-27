"""
Tools for analyzing and inspecting code.
"""

import logging
from typing import Optional
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


@tool
def get_feature_transformer_code() -> Optional[str]:
    """
    Get feature transformation code ID if available.

    Returns:
        Code ID for feature transformations or None if not available
    """
    object_registry = ObjectRegistry()

    try:
        # Feature transformer code is stored with fixed ID "feature_transformations"
        code = object_registry.get(Code, "feature_transformations")
        if code:
            return "feature_transformations"
        return None
    except KeyError:
        logger.debug("Feature transformation code not found in registry")
        return None
    except Exception as e:
        logger.warning(f"⚠️ Error getting feature transformer code: {str(e)}")
        return None
