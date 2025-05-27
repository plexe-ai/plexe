"""
Model Packager Agent for creating production-ready inference code for ML models.

This agent analyzes training code and generates high-quality inference code that can be
used in production environments, ensuring proper encapsulation of model functionality.
"""

import logging
from typing import Optional, Callable

from smolagents import CodeAgent, LiteLLMModel

from plexe.config import config
from plexe.internal.common.utils.agents import get_prompt_templates
from plexe.tools.context import get_inference_context_tool
from plexe.tools.response_formatting import format_final_mlops_agent_response
from plexe.tools.validation import validate_inference_code
from plexe.tools.code_analysis import get_feature_transformer_code

logger = logging.getLogger(__name__)


class ModelPackagerAgent:
    """
    Agent for creating production-ready inference code for ML models.

    This agent analyzes the training code produced by the ModelTrainerAgent and creates
    high-quality inference code that properly encapsulates the trained model for deployment
    in production environments.
    """

    def __init__(
        self,
        model_id: str,
        tool_model_id: str,
        verbose: bool = False,
        chain_of_thought_callable: Optional[Callable] = None,
    ):
        """
        Initialize the model packager agent.

        Args:
            model_id: Model ID for the LLM to use for inference code generation
            tool_model_id: Model ID for the LLM to use for tool operations
            verbose: Whether to display detailed agent logs
            chain_of_thought_callable: Optional callback for chain-of-thought logging
        """
        # Set verbosity level
        self.verbosity = 1 if verbose else 0

        # Create predictor builder agent - creates inference code
        self.agent = CodeAgent(
            name="MLOperationsEngineer",
            description=(
                "Expert ML operations engineer that analyzes training code and creates high-quality production-ready "
                "inference code for ML models. To work effectively, as part of the 'task' prompt the agent STRICTLY requires:"
                "- input schema for the model"
                "- output schema for the model"
                "- the 'training code id' of the training code produced by the MLEngineer agent"
            ),
            model=LiteLLMModel(model_id=model_id),
            tools=[
                get_inference_context_tool(tool_model_id),
                validate_inference_code,
                format_final_mlops_agent_response,
                get_feature_transformer_code,
            ],
            add_base_tools=False,
            verbosity_level=self.verbosity,
            additional_authorized_imports=config.code_generation.authorized_agent_imports + ["plexe", "plexe.*"],
            prompt_templates=get_prompt_templates(
                base_template_name="code_agent.yaml", override_template_name="mlops_prompt_templates.yaml"
            ),
            planning_interval=8,
            step_callbacks=[chain_of_thought_callable],
        )
