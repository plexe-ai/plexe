"""
This module defines the FeatureTransformer interface, which all generated feature transformers must implement.
"""

from abc import ABC, abstractmethod

import pandas as pd


class FeatureTransformer(ABC):
    """
    Abstract base class for all dynamically generated feature transformers.

    Every implementation of feature transformer must provide a mechanism for transforming the input data
    into the format required by the model.
    """

    @abstractmethod
    def transform(self, inputs: pd.DataFrame) -> pd.DataFrame:
        pass
