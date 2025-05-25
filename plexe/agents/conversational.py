"""
Conversational Agent for guiding users through ML model definition and initiation.

This module defines a ConversationalAgent that helps users define their ML requirements
through natural conversation, validates their inputs, and initiates model building
when all necessary information has been gathered.
"""

import logging

from smolagents import ToolCallingAgent, LiteLLMModel

from plexe.internal.common.utils.agents import get_prompt_templates
from plexe.tools.datasets import get_dataset_preview
from plexe.tools.conversation import validate_dataset_files, initiate_model_build

logger = logging.getLogger(__name__)


class ConversationalAgent:
    """
    Agent for conversational model definition and build initiation.

    This agent guides users through defining their ML requirements via natural
    conversation, helps clarify the problem, validates dataset availability,
    and initiates the model building process when all requirements are met.
    """

    def __init__(
        self,
        model_id: str = "anthropic/claude-sonnet-4-20250514",
        verbose: bool = False,
    ):
        """
        Initialize the conversational agent.

        Args:
            model_id: Model ID for the LLM to use for conversation
            verbose: Whether to display detailed agent logs
        """
        self.model_id = model_id
        self.verbose = verbose

        # Set verbosity level
        self.verbosity = 1 if verbose else 0

        # Create the conversational agent with necessary tools
        self.agent = ToolCallingAgent(
            name="ModelDefinitionAssistant",
            description=(
                "Expert ML consultant that helps users define their machine learning requirements "
                "through conversational guidance. Specializes in clarifying problem definitions, "
                "understanding data requirements, and initiating model builds when ready. "
                "Maintains a friendly, helpful conversation while ensuring all technical "
                "requirements are properly defined before proceeding with model creation."
            ),
            model=LiteLLMModel(model_id=self.model_id),
            tools=[
                get_dataset_preview,
                validate_dataset_files,
                initiate_model_build,
            ],
            add_base_tools=False,
            verbosity_level=self.verbosity,
            prompt_templates=get_prompt_templates(
                base_template_name="toolcalling_agent.yaml",
                override_template_name="conversational_prompt_templates.yaml",
            ),
        )
