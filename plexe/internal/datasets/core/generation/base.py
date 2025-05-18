"""
This module defines the base class for data generators used in the project.

Classes:
    BaseDataGenerator: Abstract base class for generating data samples in a given schema.
"""

from typing import Type, Optional

import pandas as pd
from pydantic import BaseModel
from abc import ABC, abstractmethod


class BaseDataGenerator(ABC):
    """
    Abstract base class for an object that generates data samples in a given schema.

    The BaseDataGenerator interface defines the contract for data generation operations:
    - Creating new datasets from scratch with a specified schema
    - Adding rows to existing datasets
    - Transforming existing datasets to include new columns from an extended schema
    """

    @abstractmethod
    def generate(
        self, intent: str, n_generate: int, schema: Type[BaseModel], existing_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic data for a given problem description.

        This method supports three key use cases:
        1. Create new dataset from scratch: n_generate > 0, existing_data=None
        2. Add rows to existing dataset: n_generate > 0, existing_data provided
        3. Add columns to existing dataset: n_generate=0, existing_data provided
           (transforms existing data to match new schema with additional columns)

        :param intent: Natural language description of the problem/dataset
        :param n_generate: Number of records to generate (0 for column-only generation)
        :param schema: The schema definition for the data to generate
        :param existing_data: Optional existing data to augment with new rows or columns
        :return: A pandas DataFrame containing the generated or augmented data
        """
        pass
