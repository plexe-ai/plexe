"""
Model Trainer Agent for training ML models based on provided plans.

This agent implements the training code, validates it, and executes the training code.
"""

import logging

from smolagents import CodeAgent, LiteLLMModel

from plexe.config import config
from plexe.internal.common.utils.agents import get_prompt_templates
from plexe.tools.execution import get_executor_tool
from plexe.tools.response_formatting import format_final_mle_agent_response
from plexe.tools.schemas import get_dataset_schema, get_model_schemas
from plexe.tools.training import get_training_code_generation_tool, get_training_code_fixing_tool
from plexe.tools.validation import validate_training_code
from plexe.tools.datasets import get_training_datasets
from plexe.tools.code_analysis import get_feature_transformer_code

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
        self.agent = CodeAgent(
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
            max_steps=10,
            tools=[
                get_training_code_generation_tool(tool_model_id),
                validate_training_code,
                get_dataset_schema,
                get_training_code_fixing_tool(tool_model_id),
                get_executor_tool(distributed),
                format_final_mle_agent_response,
                get_training_datasets,
                get_model_schemas,
                get_feature_transformer_code,
            ],
            add_base_tools=False,
            additional_authorized_imports=[
                "plexe",
                "plexe.*",
            ]
            + config.code_generation.authorized_agent_imports,
            verbosity_level=self.verbosity,
            prompt_templates=get_prompt_templates(
                base_template_name="code_agent.yaml", override_template_name="mle_prompt_templates.yaml"
            ),
            step_callbacks=[chain_of_thought_callable],
        )
