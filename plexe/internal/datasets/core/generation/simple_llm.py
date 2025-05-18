import asyncio
import math
import logging
from typing import Type, List, Tuple, Optional
import traceback

import pandas as pd
from pydantic import BaseModel

from plexe.internal.common.provider import Provider
from plexe.internal.common.utils.response import json_to_dataframe
from .base import BaseDataGenerator

logger = logging.getLogger(__name__)


class SimpleLLMDataGenerator(BaseDataGenerator):
    """
    Implementation of BaseDataGenerator that uses a straightforward LLM prompting mechanism to generate
    synthetic data. The generator relies on LLM inference to create or augment datasets.
    """

    def __init__(self, provider: Provider = None):
        """
        Initialize the SimpleLLMDataGenerator.

        :param provider: The provider to use for LLM queries.
        """
        from ...config import Config

        self.llm = provider
        config = Config()
        self.system_instruction = config.BASE_INSTRUCTION + config.GENERATOR_INSTRUCTION
        self.batch_size = config.BATCH_SIZE
        self.max_retries = 3

    def generate(
        self, intent: str, n_generate: int, schema: Type[BaseModel], existing_data: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Generate synthetic data based on the given intent, schema, and optionally existing data.

        :param intent: Description of the data to generate
        :param n_generate: Number of samples to generate
        :param schema: Pydantic schema defining data structure
        :param existing_data: Optional existing data to augment
        :return: DataFrame containing generated data
        """
        if n_generate == 0 and (existing_data is None or len(existing_data) == 0):
            logger.warning("No samples to generate and no existing data provided.")
            return pd.DataFrame(columns=schema.model_fields.keys())

        # Create base dataframe to store results
        columns = existing_data.columns if existing_data is not None else schema.model_fields.keys()
        df_generated = pd.DataFrame(columns=columns)

        # Create base prompt with problem specification
        base_prompt = (
            f"Give me a dataset of samples for the following ML problem:\n\n"
            f"PROBLEM DESCRIPTION:\n{intent}\n\n"
            f"SCHEMA:\n{schema.model_fields}\n\n"
        )

        # Prepare prompts for generating new data in batches
        prompts = self._prepare_generation_prompts(base_prompt, n_generate, existing_data)

        # Use asyncio to process all batches
        df_generated = asyncio.run(self._process_generation_batches(prompts, schema, df_generated))

        return df_generated

    def _prepare_generation_prompts(
        self, base_prompt: str, n_generate: int, existing_data: Optional[pd.DataFrame]
    ) -> List[Tuple[str, int]]:
        """
        Prepare prompts for batch generation.

        :param base_prompt: Base prompt with problem description and schema
        :param n_generate: Total number of samples to generate
        :param existing_data: Optional existing data to use as examples
        :return: List of (prompt, batch_size) tuples
        """
        prompts = []
        records_left = n_generate
        num_batches = math.ceil(n_generate / self.batch_size)

        for _ in range(num_batches):
            n_generate_this_iteration = min(records_left, self.batch_size)
            records_left -= n_generate_this_iteration

            # Add sample data to the prompt if available
            sample_str = ""
            if existing_data is not None and len(existing_data) > 0:
                num_samples = min(5, len(existing_data))
                sample_str = existing_data.sample(num_samples).to_string()

            prompt = (
                f"{base_prompt}"
                f"SAMPLE DATA:{sample_str}\n\n"
                f"Please give me samples that match the schema and are relevant to solving the problem. "
                f"The data should have an appropriate amount of variance and be representative of the problem. "
                f"The data should be distributed in a way that is consistent with the problem domain. "
                f"Make absolutely sure to give me EXACTLY {n_generate_this_iteration} records. "
                f"You must give me no fewer than and no more than {n_generate_this_iteration} records. "
                f"In your response, only include the dataset as a JSON string, no other text. "
                f"The output must be a raw JSON string with no formatting characters. "
                f"Do not give me any code, any descriptions, any explanations, or any other text of any kind. "
                f"Only give me a raw JSON string with the data, and no other information whatsoever."
            )
            prompts.append((prompt, n_generate_this_iteration))

        return prompts

    async def _process_generation_batches(
        self, prompts: List[Tuple[str, int]], schema: Type[BaseModel], df_generated: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Process all generation batches asynchronously with retry logic.

        :param prompts: List of (prompt, batch_size) tuples
        :param schema: Pydantic schema for validation
        :param df_generated: DataFrame to store results
        :return: DataFrame with all generated data
        """
        pending_prompts = prompts.copy()
        retry_count = 0

        while pending_prompts and retry_count < self.max_retries:
            logger.info(
                f"Generating data for {len(pending_prompts)} batches (attempt {retry_count + 1}/{self.max_retries})..."
            )

            # Process all pending prompts in parallel
            tasks = [self._generate_batch(prompt, schema) for prompt, _ in pending_prompts]
            batch_results = await asyncio.gather(*tasks)

            # Process results and collect failed prompts for retry
            failed_prompts = []
            for result, (prompt, n_expected) in zip(batch_results, pending_prompts):
                if result is not None:
                    df_generated = pd.concat([df_generated, result], ignore_index=True)
                    logger.debug(f"Successfully generated {len(result)} samples")
                else:
                    failed_prompts.append((prompt, n_expected))

            # Update for next iteration
            pending_prompts = failed_prompts
            if failed_prompts:
                logger.warning(f"Retrying {len(failed_prompts)} failed batches...")
                retry_count += 1
            else:
                logger.info("All batches processed successfully.")
                break

        if pending_prompts:
            logger.error(
                f"Failed to generate {sum(n for _, n in pending_prompts)} samples after {self.max_retries} attempts"
            )

        return df_generated

    async def _generate_batch(self, prompt: str, schema: Type[BaseModel]) -> Optional[pd.DataFrame]:
        """
        Generate a single batch of data asynchronously.

        :param prompt: The generation prompt
        :param schema: Pydantic schema for validation
        :return: DataFrame with generated data or None if failed
        """
        try:
            response = await asyncio.to_thread(self.llm.query, self.system_instruction, prompt, schema)

            if response is None:
                logger.error("Received None response from LLM")
                return None

            # Use the utility function to convert JSON to DataFrame
            try:
                df_batch = json_to_dataframe(response)

                # Validate that all required schema fields are present
                missing_fields = [field for field in schema.model_fields.keys() if field not in df_batch.columns]
                if missing_fields:
                    logger.error(f"Generated data missing required fields: {missing_fields}")
                    return None

            except ValueError as e:
                logger.error(f"Failed to parse LLM response: {str(e)}")
                return None

            return df_batch

        except Exception as e:
            logger.error(f"Error during batch generation: {str(e)}")
            logger.debug(traceback.format_exc())
            return None
