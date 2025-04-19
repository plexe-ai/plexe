"""
This module defines the base class for data generators used in the project.

Classes:
    BaseDataGenerator: Abstract base class for generating data samples in a given schema.
"""

from typing import Type

import pandas as pd
from pydantic import BaseModel
from abc import ABC, abstractmethod


class BaseDataGenerator(ABC):
    """
    Abstract base class for an object that generates data samples in a given schema.
    """

    @abstractmethod
    def generate(
        self, intent: str, n_generate: int, schema: Type[BaseModel], existing_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate synthetic data for a given problem description.
        :param intent: natural language description of the problem
        :param n_generate: number of records to generate
        :param schema: the schema of the data to generate
        :param existing_data: existing data to augment
        :return: a pandas DataFrame containing the generated data
        """
        pass
