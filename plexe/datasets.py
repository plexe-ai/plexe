"""
This module provides the Dataset class, which represents a collection of data that can be real,
synthetic, or a combination of both.

The Dataset class offers functionalities for:
- Wrapping existing datasets (e.g. pandas DataFrames).
- Generating synthetic data based on a schema.
- Augmenting real datasets with additional synthetic samples.
- Iterating and accessing data samples conveniently.

Users can either pass raw datasets directly to models or leverage this class for dataset management and augmentation.
"""

from typing import Iterator, Type, Dict, Optional
import logging
import pandas as pd
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from plexe.internal.common.datasets.interface import TabularConvertible
from plexe.internal.common.provider import Provider
from plexe.internal.common.datasets.adapter import DatasetAdapter
from plexe.internal.common.utils.pydantic_utils import merge_models, map_to_basemodel
from plexe.internal.schemas.resolver import SchemaResolver
from plexe.internal.datasets.generator import DatasetGenerator as DataGenerator


class DatasetGenerator:
    """
    Represents a dataset, which can contain real data, synthetic data, or both.

    This class provides a structured way to manage data, allowing users to:
    - Wrap real datasets (pandas etc.).
    - Generate synthetic data from scratch.
    - Augment existing datasets with synthetic samples.
    - Add new columns to existing datasets using an extended schema.

    Example:
        >>> synthetic_dataset = DatasetGenerator(
        >>>     description="Synthetic reviews",
        >>>     provider="openai/gpt-4o",
        >>>     schema=MovieReviewSchema,
        >>> )
        >>> synthetic_dataset.generate(100)  # Generate 100 samples
        >>> model.build(datasets={"train": synthetic_dataset})
    """

    def __init__(
        self,
        description: str,
        provider: str,
        schema: Type[BaseModel] | Dict[str, type] = None,
        data: pd.DataFrame = None,
    ) -> None:
        """
        Initialize a new DatasetGenerator.

        :param description: A human-readable description of the dataset
        :param provider: LLM provider used for synthetic data generation
        :param schema: The schema the data should match, if any
        :param data: A dataset of real data on which to base the generation, if available
        """
        # Core attributes required for dataset generation
        self.description = description
        self.provider = Provider(provider)

        # Internal attributes for data management
        self._data: Optional[pd.DataFrame] = None
        self._index = 0
        self.schema = None

        # Process schema and data inputs
        if schema is not None:
            # Convert schema to Pydantic BaseModel if it's a dictionary
            self.schema = map_to_basemodel("data", schema)

        if data is not None:
            # Convert and validate input data
            data_wrapper = DatasetAdapter.coerce(data)
            if isinstance(data_wrapper, TabularConvertible):
                self._data = data_wrapper.to_pandas()
            else:
                raise ValueError("Dataset must be convertible to pandas DataFrame.")

            # If schema is provided, validate data against schema
            # but only validate existing columns, not new ones being added
            if schema is not None:
                self._validate_schema(self._data, allow_new_columns=True)
            # If no schema provided, infer it from data
            else:
                schemas = SchemaResolver(self.provider, self.description).resolve({"data": self._data})
                self.schema = merge_models("data", list(schemas))

        # Initialize data generator
        self.data_generator = DataGenerator(self.provider, self.description, self.schema)

    def generate(self, num_samples: int):
        """
        Generate synthetic data samples or augment existing data.

        If num_samples is 0 and existing data is provided with a new schema,
        this will transform the existing data to match the new schema (adding columns).

        :param num_samples: Number of new samples to generate
        """
        generated_data = self.data_generator.generate(num_samples, self._data)

        if self._data is None:
            self._data = generated_data
        elif num_samples == 0:
            # When num_samples is 0, we're just adding columns to existing data
            # SimpleLLMDataGenerator.generate already handles this correctly by returning
            # the existing data with new columns added, so we just replace _data directly
            self._data = generated_data
        else:
            # When adding new rows, concatenate them with existing data
            self._data = pd.concat([self._data, generated_data], ignore_index=True)

    def _validate_schema(self, data: pd.DataFrame, allow_new_columns: bool = False):
        """
        Ensure data matches the schema by checking column presence.

        :param data: DataFrame to validate against the schema
        :param allow_new_columns: If True, allow schema to have columns that don't exist in data yet
        :raises ValueError: If required columns from schema are missing and not allowed
        """
        for key in self.schema.model_fields.keys():
            if key not in data.columns:
                if not allow_new_columns:
                    raise ValueError(f"Dataset does not match schema, missing column in dataset: {key}")
                else:
                    # When augmenting with new columns, we'll skip validation for those columns
                    logger.debug(f"Allowing new column that will be added through augmentation: {key}")

    @property
    def data(self) -> pd.DataFrame:
        """
        Get the dataset as a pandas DataFrame.

        :return: The dataset as a DataFrame
        :raises ValueError: If no data has been set or generated
        """
        if self._data is None:
            raise ValueError("No data has been set or generated. Call generate() first.")
        return self._data

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        :return: Number of rows in the dataset, or 0 if no data
        """
        if self._data is not None:
            return len(self._data)
        return 0

    def __iter__(self) -> Iterator:
        """
        Get an iterator over the dataset rows.

        :return: Self as iterator
        """
        self._index = 0
        return self

    def __next__(self):
        """
        Get the next item when iterating over the dataset.

        :return: Dictionary representing the next row
        :raises StopIteration: When all rows have been processed
        """
        if self._data is None or self._index >= len(self):
            raise StopIteration

        row = self._data.iloc[self._index].to_dict()
        self._index += 1
        return row

    def __getitem__(self, index: int):
        """
        Get a dataset item by index.

        :param index: Row index to retrieve
        :return: Dictionary representing the row at the given index
        :raises IndexError: If dataset is empty
        """
        if self._data is None:
            raise IndexError("Dataset is empty.")
        return self._data.iloc[index].to_dict()
