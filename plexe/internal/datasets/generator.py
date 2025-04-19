from pydantic import BaseModel
from typing import Type

import pandas as pd

from plexe.internal.common.provider import Provider


class DatasetGenerator:
    """Generate synthetic data based on request parameters."""

    def __init__(self, provider: Provider, description: str, schema: Type[BaseModel]):
        """
        Initialize the DatasetGenerator with a provider.

        :param [Provider] provider: The provider to use for data generation.
        :param [str] description: The description of the data to generate.
        :param [Type[BaseModel]] schema: The schema for the data to generate.
        """
        from .core.generation.combined import CombinedDataGenerator
        from .config import Config

        self.provider = provider
        self.description = description
        self.schema = schema
        self.generator = CombinedDataGenerator(self.provider, Config())

    def generate(self, n_samples: int, existing_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generate synthetic data based on request parameters.

        :param [int] n_samples: The number of samples to generate.
        :param [SupportedDatasets] existing_data: The existing data to augment.
        :return [SupportedDatasets]: The generated synthetic data.
        """

        return self.generator.generate(
            intent=self.description,
            n_generate=n_samples,
            schema=self.schema,
            existing_data=existing_data,
        )
