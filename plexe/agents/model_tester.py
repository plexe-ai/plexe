"""
Model Tester Agent for comprehensive testing and evaluation of finalized ML models.

This module defines a ModelTesterAgent that evaluates model performance on test data,
performs quality analysis, and generates comprehensive evaluation reports.
"""

import logging
from typing import Optional, Callable

from smolagents import CodeAgent, LiteLLMModel

from plexe.config import config
from plexe.internal.common.utils.agents import get_prompt_templates
from plexe.tools.testing import register_testing_code, register_evaluation_report
from plexe.tools.datasets import get_test_dataset
from plexe.tools.schemas import get_model_schemas
from plexe.tools.code_analysis import get_feature_transformer_code

logger = logging.getLogger(__name__)


class ModelTesterAgent:
    """
    Agent for comprehensive testing and evaluation of finalized ML models.

    This agent retrieves the predictor and test datasets directly from the object
    registry, performs thorough evaluation through direct code execution, and
    produces detailed performance reports.
    """

    def __init__(
        self,
        model_id: str,
        verbose: bool = False,
        chain_of_thought_callable: Optional[Callable] = None,
    ):
        """
        Initialize the model tester agent.

        Args:
            model_id: Model ID for the LLM to use for model testing
            verbose: Whether to display detailed agent logs
            chain_of_thought_callable: Optional callable for chain of thought logging
        """
        self.model_id = model_id
        self.verbose = verbose

        # Set verbosity level
        self.verbosity = 1 if verbose else 0

        # Create the model tester agent with the necessary tools
        self.agent = CodeAgent(
            name="ModelTester",
            description=(
                "Expert ML model testing specialist that evaluates finalized models for performance, "
                "quality, and production readiness. To work effectively, as part of the 'task' prompt "
                "the agent STRICTLY requires:\n"
                "- the test dataset name for evaluation\n"
                "- task definition and target metric information\n"
                "- the expected input schema and output schema of the model\n"
                "The predictor must already have been created by the ml operations engineer.\n"
            ),
            model=LiteLLMModel(model_id=self.model_id),
            tools=[
                register_testing_code,
                register_evaluation_report,
                get_test_dataset,
                get_model_schemas,
                get_feature_transformer_code,
            ],
            add_base_tools=False,
            verbosity_level=self.verbosity,
            additional_authorized_imports=config.code_generation.authorized_agent_imports
            + [
                "plexe",
                "plexe.*",
                "pandas",
                "pandas.*",
                "numpy",
                "numpy.*",
                "sklearn",
                "sklearn.*",
                "scipy",
                "scipy.*",
            ],
            prompt_templates=get_prompt_templates(
                base_template_name="code_agent.yaml", override_template_name="model_tester_prompt_templates.yaml"
            ),
            step_callbacks=[chain_of_thought_callable],
        )
