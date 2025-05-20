"""
Model state definitions for Plexe.

This module defines the possible states a model can be in during its lifecycle.
"""

from enum import Enum


class ModelState(Enum):
    """States a model can be in during its lifecycle."""

    DRAFT = "draft"
    """Model is in draft state, not yet built."""

    BUILDING = "building"
    """Model is currently being built."""

    READY = "ready"
    """Model is built and ready to use."""

    ERROR = "error"
    """Model encountered an error during building."""
