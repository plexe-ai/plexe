"""
This module provides a data generator implementation that combines multiple generation techniques.
"""

import abc
import json
import logging
import math
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Type, List

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from plexe.internal.common.provider import Provider
from .base import BaseDataGenerator
from ...config import Config

# configure
logger = logging.getLogger(__name__)


class CombinedDataGenerator(BaseDataGenerator):
    """
    Implementation of BaseDataGenerator that combines all the data generation techniques implemented in this
    project. This generator is intended to be used as the 'primary' production data generator implementation
    in projects.
    """

    def __init__(self, provider: Provider, config: Config):
        self.tp_executor = ThreadPoolExecutor(max_workers=100)
        self.max_n_llm_samples = config.MAX_N_LLM_SAMPLES
        self.batch_size = config.BATCH_SIZE
        self.provider = provider

        self.generator = self.GeneratorModel(provider, 20000, config.BASE_INSTRUCTION + config.GENERATOR_INSTRUCTION)
        self.filter = self.FilterModel(provider, 20000, config.BASE_INSTRUCTION + config.FILTER_INSTRUCTION)
        self.labeller = self.LabellerModel(provider, 20000, config.BASE_INSTRUCTION + config.LABELLER_INSTRUCTION)
        self.reviewer = self.ReviewerModel(provider, 20000, config.BASE_INSTRUCTION + config.REVIEWER_INSTRUCTION)

    def generate(
        self, intent: str, n_generate: int, schema: Type[BaseModel], existing_data: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Generate synthetic data based on problem description and schema
        """
        try:
            return self._generate_dataset(n_generate, intent, schema, existing_data)
        except Exception as e:
            logger.error(f"Data generation failed: {e}")
            raise

    def _generate_dataset(
        self, n_to_generate: int, description: str, schema: Type[BaseModel], sample_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Generate complete dataset"""
        # Initialize empty dataframe
        columns = schema.model_fields.keys()
        df_generated = pd.DataFrame(columns=columns)

        num_batches = math.ceil(n_to_generate / self.batch_size)
        logger.info(f"Generating {n_to_generate} samples in {num_batches} batches")

        overall_pbar = tqdm(total=n_to_generate, desc="Total samples generated", unit="samples")

        # Generate batches
        with self.tp_executor as executor:
            futures = []
            for i in range(num_batches):
                batch_size = min(self.batch_size, n_to_generate - i * self.batch_size)
                sample_data = None
                if sample_df is not None and len(sample_df) > 0:
                    sample_data = sample_df.sample(n=min(20, len(sample_df)), replace=True)

                futures.append(executor.submit(self._generate_batch, batch_size, description, schema, sample_data))

            for future in as_completed(futures):
                try:
                    batch_df = future.result()
                    if batch_df is not None and len(batch_df) > 0:
                        df_generated = pd.concat([df_generated, batch_df], ignore_index=True)
                        overall_pbar.update(len(batch_df))
                except Exception as e:
                    logger.error(f"Failed to process batch: {e}")

        overall_pbar.close()

        if len(df_generated) == 0:
            raise RuntimeError("Failed to generate any valid data")

        return df_generated

    def _generate_batch(
        self, batch_size: int, description: str, schema: Type[BaseModel], sample_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Generate a batch of data"""
        try:
            batch_df = self.generator.generate(
                n_to_generate=batch_size, description=description, schema=schema, reference_df=sample_df
            )
            return batch_df

        except Exception as e:
            logger.error(f"Batch generation failed: {str(e)}")
            return pd.DataFrame(columns=schema.model_fields.keys())

    class Model(abc.ABC):
        def __init__(self, provider: Provider):
            self.llm: Provider = provider

    class GeneratorModel(Model):
        def __init__(self, provider, max_tokens: int, instruction):
            super().__init__(provider)
            self.max_tokens = max_tokens
            self.instruction = instruction

        def generate(self, n_to_generate, description, schema: Type[BaseModel], reference_df=None) -> pd.DataFrame:
            # sample reference data, if available
            if reference_df is not None:
                sample_data_str = f"SAMPLE DATA:\n{reference_df.to_string()}\n\n"
            else:
                sample_data_str = ""
            # generate the prompt
            prompt = (
                f"Generate {n_to_generate} samples for the following ML problem:\n\n{description}\n\n"
                f"SCHEMA:\n{schema.model_fields}\n\n"
                f"{sample_data_str}"
                f"Ensure you generate exactly {n_to_generate} records in the specified format. "
            )
            logger.debug(prompt)

            class DataResponseFormat(BaseModel):
                records: List[schema]

            # generate the content
            response = json.loads(self.llm.query(self.instruction, prompt, response_format=DataResponseFormat))
            logger.debug(response)
            # return as dataframe
            return pd.DataFrame().from_dict(response["records"]).dropna()

    class FilterModel(Model):
        def __init__(self, provider, max_tokens: int, instruction):
            super().__init__(provider)
            self.max_tokens = max_tokens
            self.instruction = instruction

        def filter(self, description, schema, df: pd.DataFrame) -> pd.DataFrame:
            prompt = (
                f"Filter samples for the following ML problem:\n\n{description}\n\n"
                f"SCHEMA:\n{schema}\n\n"
                f"DATASET TO FILTER:\n{df.to_json(orient='records')}\n\n"
                "Return the dataset as raw JSON, without additional text."
            )
            logger.debug(prompt)
            r = self.llm.query(self.instruction, prompt, schema)
            logger.debug(r)
            return json_to_df(r)

    class LabellerModel(Model):
        def __init__(self, provider, max_tokens: int, instruction):
            super().__init__(provider)
            self.max_tokens = max_tokens
            self.instruction = instruction

        def label(self, description, schema, df: pd.DataFrame) -> pd.DataFrame:
            prompt = (
                f"Label each sample for the ML problem:\n\n{description}\n\n"
                f"SCHEMA:\n{schema}\n\n"
                f"DATASET TO LABEL:\n{df.to_json(orient='records')}\n\n"
                "Add a column matching the target variable's name with the labels. Return as raw JSON."
            )
            logger.debug(prompt)
            r = self.llm.query(self.instruction, prompt, schema)
            logger.debug(r)
            return json_to_df(r)

    class ReviewerModel(Model):
        def __init__(self, provider, max_tokens: int, instruction):
            super().__init__(provider)
            self.max_tokens = max_tokens
            self.instruction = instruction

        def review(self, description, schema, df: pd.DataFrame) -> pd.DataFrame:
            prompt = (
                f"Review the batch for the ML problem:\n\n{description}\n\n"
                f"SCHEMA:\n{schema}\n\n"
                f"DATASET TO REVIEW:\n{df.to_json(orient='records')}\n\n"
                "Return the dataset as raw JSON with 'removal' and 'removal_reason' columns."
            )
            logger.debug(prompt)
            r = self.llm.query(self.instruction, prompt, schema)
            logger.debug(r)
            logger.debug(r)
            batch = json_to_df(self.llm.query(self.instruction, prompt))
            batch = batch[batch["removal"] == 0].drop(columns=["removal", "removal_reason"])
            logger.debug(batch)
            return batch


def json_to_df(json_str: str, handle_partial: bool = True) -> pd.DataFrame:
    """
    Convert a JSON string to a DataFrame.
    :param json_str:
    :param handle_partial:
    :return:
    """
    try:
        logger.debug(f"Converting JSON to DataFrame:\n{json_str}")

        # Remove code blocks
        if "```" in json_str:
            import re

            json_str = re.search(r"```(?:json)?\s*(.*?)\s*```", json_str, re.DOTALL)
            if json_str:
                json_str = json_str.group(1)

        # Clean up the string
        json_str = json_str.strip()

        # Find the actual JSON array
        start_idx = json_str.find("[")
        end_idx = json_str.rfind("]")
        if start_idx >= 0 and end_idx >= 0:
            json_str = json_str[start_idx : end_idx + 1]

        json_str = json_str.replace("\n", " ").replace("\r", " ")

        # Remove apostrophes and handle quotes for JSON
        json_str = json_str.replace("'", "")
        json_str = json_str.replace('"', '\\"').replace('\\"', '"')

        import json

        data = json.loads(json_str)
        df = pd.DataFrame(data)
        logger.debug("Successfully converted JSON to DataFrame")
        return df

    except Exception as e:
        logger.error(f"Error converting JSON to DataFrame: {e}")
        logger.debug(f"JSON string:\n{json_str}")
        return pd.DataFrame()


def load_sample_data(sample_data_path, schema):
    # Load sample data, if available
    if sample_data_path is not None:
        sample_df = pd.read_csv(sample_data_path)
        columns_to_keep = list(schema["column_names"])
        # Drop columns from sample_df that are not mentioned in schema
        sample_df = sample_df[sample_df.columns.intersection(columns_to_keep)]
        logger.debug(f"Loaded sample data with schema:\n{sample_df.dtypes}")
    else:
        sample_df = None
        logger.debug("No sample data provided.")
    return sample_df
