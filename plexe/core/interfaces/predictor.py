"""
This module defines the Predictor interface, which all dynamically generated inference codes must implement.
"""

from abc import ABC, abstractmethod
from typing import List
from plexe.internal.models.entities.artifact import Artifact


class Predictor(ABC):
    """
    Abstract base class for all dynamically generated inference code.

    Every implementation of predictor must provide a mechanism for instantiating the underlying model(s)
    by reading the binary or text data from the `ModelArtifact` class.
    """

    @abstractmethod
    def __init__(self, artifacts: List[Artifact]):
        pass

    @abstractmethod
    def predict(self, inputs: dict) -> dict:
        pass
