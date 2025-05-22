"""
Schema Resolver Agent for inferring input and output schemas for ML models.

This module defines a SchemaResolverAgent that determines the appropriate
input and output schemas for a machine learning model based on its intent
and available datasets.
"""

import logging
from typing import Callable

from smolagents import LiteLLMModel, CodeAgent

from plexe.internal.common.utils.agents import get_prompt_templates
from plexe.tools.datasets import get_dataset_preview, get_eda_report
from plexe.tools.schemas import register_final_model_schemas

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
        model_id: str,
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
        self.agent = CodeAgent(
            name="SchemaResolver",
            description=(
                "Expert schema resolver that determines appropriate input and output schemas for ML models. "
                "To work effectively, as part of the 'task' prompt the agent STRICTLY requires:\n"
                "- the ML task definition (i.e. 'intent')\n"
                "- the NAME of the dataset to be used for training\n"
                "- input schema already defined for this task, if available\n"
                "- output schema already defined for this task, if available\n"
            ),
            model=LiteLLMModel(model_id=self.model_id),
            tools=[get_dataset_preview, get_eda_report, register_final_model_schemas],
            add_base_tools=False,
            verbosity_level=self.verbosity,
            step_callbacks=[chain_of_thought_callable],
            prompt_templates=get_prompt_templates(
                base_template_name="code_agent.yaml", override_template_name="schema_resolver_prompt_templates.yaml"
            ),
        )
