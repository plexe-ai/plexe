"""
This module provides utility functions for working with model descriptions and metadata.
"""

from typing import Optional


def calculate_model_size(artifacts: list) -> Optional[int]:
    """
    Calculate the total size of the model artifacts in bytes.

    :param artifacts: List of artifacts with path attributes
    :return: The size in bytes or None if no artifacts exist
    """
    if not artifacts:
        return None

    total_size = 0
    for artifact in artifacts:
        if artifact.path and artifact.path.exists():
            total_size += artifact.path.stat().st_size

    return total_size if total_size > 0 else None


def format_code_snippet(code: Optional[str]) -> Optional[str]:
    """
    Format a code snippet for display, truncating if necessary.

    :param code: The source code as a string
    :return: A formatted code snippet or None if code doesn't exist
    """
    if not code:
        return None

    # Limit the size of code displayed, possibly add line numbers, etc.
    lines = code.splitlines()
    if len(lines) > 20:
        # Return first 10 and last 10 lines with a note in the middle
        return "\n".join(lines[:10] + ["# ... additional lines omitted ..."] + lines[-10:])
    return code
