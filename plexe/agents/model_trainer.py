"""
Model Trainer Agent for training ML models based on provided plans.

This agent implements the training code, validates it, and executes the training code.
"""

import logging

from smolagents import ToolCallingAgent, LiteLLMModel

from plexe.internal.common.utils.agents import get_prompt_templates
from plexe.internal.models.tools.execution import get_executor_tool
from plexe.internal.models.tools.response_formatting import format_final_mle_agent_response
from plexe.internal.models.tools.training import get_training_code_generation_tool, get_training_code_fixing_tool
from plexe.internal.models.tools.validation import validate_training_code

logger = logging.getLogger(__name__)


class ModelTrainerAgent:
    """
    Agent for training ML models based on provided plans.

    This agent implements the training code, validates it, and executes the training code.
    """

    def __init__(
        self,
        ml_engineer_model_id: str,
        tool_model_id: str,
        distributed: bool = False,
        verbose: bool = False,
        chain_of_thought_callable: callable = None,
    ):
        # Set verbosity level
        self.verbosity = 1 if verbose else 0

        # Create model trainer agent - implements training code
        self.agent = ToolCallingAgent(
            name="MLEngineer",
            description=(
                "Expert ML engineer that implements, trains and validates ML models based on provided plans. "
                "To work effectively, as part of the 'task' prompt the agent STRICTLY requires:"
                "- the ML task definition (i.e. 'intent')"
                "- input schema for the model"
                "- output schema for the model"
                "- the name and comparison method of the metric to optimise"
                "- the full solution plan that outlines how to solve this problem"
                "- the split train/validation dataset names"
                "- the working directory to use for model execution"
            ),
            model=LiteLLMModel(model_id=ml_engineer_model_id),
            tools=[
                get_training_code_generation_tool(tool_model_id),
                validate_training_code,
                get_training_code_fixing_tool(tool_model_id),
                get_executor_tool(distributed),
                format_final_mle_agent_response,
            ],
            add_base_tools=False,
            verbosity_level=self.verbosity,
            prompt_templates=get_prompt_templates("toolcalling_agent.yaml", "mle_prompt_templates.yaml"),
            step_callbacks=[chain_of_thought_callable],
        )
