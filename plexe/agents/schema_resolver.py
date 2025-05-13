"""
Schema Resolver Agent for inferring input and output schemas for ML models.

This module defines a SchemaResolverAgent that determines the appropriate
input and output schemas for a machine learning model based on its intent
and available datasets.
"""

import json
import logging
from typing import Dict, List, Any, Callable

from smolagents import ToolCallingAgent, LiteLLMModel

from plexe.config import prompt_templates
from plexe.internal.common.registries.objects import ObjectRegistry
from plexe.internal.models.tools.datasets import get_dataset_preview
from plexe.internal.models.tools.schemas import get_raw_dataset_schema, get_model_schema_resolver

logger = logging.getLogger(__name__)


class SchemaResolverAgent:
    """
    Agent for resolving input and output schemas for ML models.

    This agent analyzes the model intent and available datasets to determine
    the appropriate input and output schemas, handling both schema inference
    and validation scenarios.
    """

    def __init__(
        self,
        model_id: str = "openai/gpt-4o",
        verbose: bool = False,
        chain_of_thought_callable: Callable = None,
    ):
        """
        Initialize the schema resolver agent.

        Args:
            model_id: Model ID for the LLM to use for schema resolution
            verbose: Whether to display detailed agent logs
        """
        self.model_id = model_id
        self.verbose = verbose

        # Set verbosity level
        self.verbosity = 1 if verbose else 0

        # Create the schema resolver agent with the necessary tools
        self.agent = ToolCallingAgent(
            name="SchemaResolver",
            description=(
                "Expert schema resolver that determines the appropriate input and output "
                "schemas for ML models based on intent and available datasets."
            ),
            model=LiteLLMModel(model_id=self.model_id),
            tools=[get_dataset_preview, get_raw_dataset_schema, get_model_schema_resolver(model_id)],
            add_base_tools=False,
            verbosity_level=self.verbosity,
            step_callbacks=chain_of_thought_callable,
        )

    def run(
        self,
        intent: str,
        dataset_names: List[str],
        user_input_schema: Dict[str, str] = None,
        user_output_schema: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """
        Run the schema resolver agent to determine input and output schemas.

        Args:
            intent: Natural language description of the model's purpose
            dataset_names: List of dataset registry names available to the model
            user_input_schema: Optional user-provided input schema
            user_output_schema: Optional user-provided output schema

        Returns:
            Dictionary containing:
            - input_schema: The finalized input schema
            - output_schema: The finalized output schema
            - reasoning: Explanation of schema design decisions
        """
        # Use the template system to create the prompt
        has_input_schema = user_input_schema is not None
        has_output_schema = user_output_schema is not None

        # Format dataset names for the template
        datasets_str = ", ".join(dataset_names)

        # Format schemas as needed
        input_schema_str = json.dumps(user_input_schema, indent=2) if user_input_schema else None
        output_schema_str = json.dumps(user_output_schema, indent=2) if user_output_schema else None

        # Generate the prompt using the template system
        task_description = prompt_templates.schema_resolver_prompt(
            intent=intent,
            datasets=datasets_str,
            input_schema=input_schema_str,
            output_schema=output_schema_str,
            has_input_schema=has_input_schema,
            has_output_schema=has_output_schema,
        )

        # Run the agent to get schema resolution
        self.agent.run(task_description)

        # The registry should already have the schemas registered by the tool
        object_registry = ObjectRegistry()

        # Ensure the registry has the schema information registered
        # (this is done by the define_model_schemas tool)
        input_schema = object_registry.get(dict, "input_schema")
        output_schema = object_registry.get(dict, "output_schema")
        schema_reasoning = object_registry.get(str, "schema_reasoning", default="No reasoning provided")

        # Return the schema information
        return {"input_schema": input_schema, "output_schema": output_schema, "reasoning": schema_reasoning}
