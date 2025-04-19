import asyncio
import math
from typing import Type

import pandas as pd
from pydantic import BaseModel

from plexe.internal.common.provider import Provider
from .base import BaseDataGenerator


class SimpleLLMDataGenerator(BaseDataGenerator):
    """
    Implementation of BaseDataGenerator that uses a straightforward LLM prompting mechanism to generate
    synthetic data. The generator relies on a single inference call to a pre-trained LLM model to generate samples.
    """

    def __init__(self, provider: Provider = None):
        self.llm = provider
        self.system_instruction = (
            "You are an expert in data science, data engineering, and any problem domain you encounter. "
            "You are speaking to someone who is, likewise, an expert in all these areas. "
            "Expectations for your performance are extremely high. Mediocrity is not acceptable. "
        )

    def generate(
        self, intent: str, n_generate: int, schema: Type[BaseModel], existing_data: pd.DataFrame = None
    ) -> pd.DataFrame:
        # basic problem specification
        base_prompt = (
            f"Give me a dataset of samples for the following ML problem:\n\n" f"PROBLEM DESCRIPTION:\n{intent}\n\n"
        )

        df_generated = pd.DataFrame(
            columns=existing_data.columns if existing_data is not None else schema.model_fields.keys()
        )

        # prepare prompts for all batches
        batch_size = 60
        num_batches = math.ceil(n_generate / batch_size)
        records_left = n_generate

        prompts = []
        for _ in range(num_batches):
            n_generate_this_iteration = min(records_left, batch_size)
            records_left -= n_generate_this_iteration

            # add sample data to the prompt if available
            sample_str = existing_data.sample(5).to_string() if existing_data is not None else ""
            prompt = (
                f"{base_prompt}"
                f"SAMPLE DATA:{sample_str}\n\n"
                f"Please give me samples that match the schema and are relevant to solving the problem. "
                f"The data should have an appropriate amount of variance and be representative of the problem. "
                f"The data should be distributed in a way that is consistent with the problem domain. "
                f"Make absolutely sure to give me EXACTLY {n_generate_this_iteration} records. "
                f"You must give me no fewer than and no more than {n_generate_this_iteration} records. "
                f"In your response, only include the dataset as a JSON string, no other text. "
                f"The output must be a raw JSON string with no formatting characters."
                f"Do not give me any code, any descriptions, any explanations, or any other text of any kind. "
                f"Only give me a raw JSON string with the data, and no other information whatsoever. "
            )
            prompts.append((prompt, n_generate_this_iteration))

        # generate data for a prompt
        async def generate_data(prompt):
            loop = asyncio.get_running_loop()
            try:
                return await loop.run_in_executor(None, self.llm.query, self.system_instruction, prompt, schema)
            except Exception as err:
                print(f"Error during generation: {err}")
                return None  # Indicate failure

        # Function to run all tasks asynchronously
        async def run_tasks(p):
            tasks = [generate_data(prompt) for prompt, _ in p]
            return await asyncio.gather(*tasks)

        # generate results asynchronously retry failed batches
        pending_prompts = prompts.copy()
        while pending_prompts:
            print(f"Generating data for {len(pending_prompts)} batches...")
            responses = asyncio.run(run_tasks(pending_prompts))

            failed_prompts = []

            for response, (prompt, n_generate_this_iteration) in zip(responses, pending_prompts):
                if response is not None:
                    try:
                        response_text = response.text.replace("json", "").replace("`", "")
                        # convert the data to pd dataframe and append to the generated data
                        df_generated = pd.concat([df_generated, pd.read_json(str(response_text))], ignore_index=True)
                    except Exception as e:
                        print(f"Error processing data: {e}")
                        # Add the prompt back to failed_prompts for retry
                        failed_prompts.append((prompt, n_generate_this_iteration))
                else:
                    # If response is None, generation failed
                    failed_prompts.append((prompt, n_generate_this_iteration))

            # Update the pending_prompts with failed batches
            pending_prompts = failed_prompts

            if failed_prompts:
                print(f"Retrying {len(failed_prompts)} failed batches...")
            else:
                print("All batches processed successfully.")

        # Parse the response and return the synthetic data
        return df_generated
