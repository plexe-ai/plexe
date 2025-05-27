"""
Feature Engineering Agent for transforming raw datasets into optimized features for ML models.

This agent analyzes datasets based on EDA reports, generates feature transformation code,
validates it, and executes it to create enhanced datasets for model training.
"""

import logging
from typing import Optional, Callable

from smolagents import CodeAgent, LiteLLMModel

from plexe.config import config
from plexe.internal.common.utils.agents import get_prompt_templates
from plexe.tools.datasets import get_dataset_preview, get_eda_reports, get_latest_datasets
from plexe.tools.execution import apply_feature_transformer
from plexe.tools.validation import validate_feature_transformations
from plexe.tools.schemas import get_model_schemas

logger = logging.getLogger(__name__)


class FeatureEngineeringAgent:
    """
    Agent for creating optimized features from raw datasets for ML models.

    This agent analyzes datasets, generates transformation code based on EDA insights,
    and applies these transformations to create enhanced datasets for model training.
    The agent ensures data integrity through validation and provides transformed
    datasets through the object registry.
    """

    def __init__(
        self,
        model_id: str,
        verbose: bool = False,
        chain_of_thought_callable: Optional[Callable] = None,
    ):
        """
        Initialize the feature engineering agent.

        Args:
            model_id: Model ID for the LLM to use for feature engineering
            verbose: Whether to display detailed agent logs
            chain_of_thought_callable: Optional callback for chain-of-thought logging
        """
        # Set verbosity level
        self.verbosity = 1 if verbose else 0

        # Create feature engineering agent
        self.agent = CodeAgent(
            name="FeatureEngineer",
            description=(
                "Expert data scientist that transforms raw datasets into optimized features for ML models. "
                "To work effectively, as part of the 'task' prompt the agent STRICTLY requires:"
                "- the ML task definition (i.e. 'intent')"
                "- the name of the dataset to transform"
            ),
            model=LiteLLMModel(model_id=model_id),
            tools=[
                get_dataset_preview,
                validate_feature_transformations,
                apply_feature_transformer,
                get_latest_datasets,
                get_eda_reports,
                get_model_schemas,
            ],
            add_base_tools=False,
            additional_authorized_imports=config.code_generation.authorized_agent_imports
            + [
                "plexe",
                "plexe.*",
                "pandas",
                "pandas.*",
                "numpy",
                "numpy.*",
                "scikit-learn",
                "scikit-learn.*",
                "scikit_learn",
                "scikit_learn.*",
                "sklearn",
                "sklearn.*",
                "scipy",
                "scipy.*",
                "statsmodels",
                "statsmodels.*",
            ],
            verbosity_level=self.verbosity,
            prompt_templates=get_prompt_templates(
                base_template_name="code_agent.yaml", override_template_name="feature_engineer_prompt_templates.yaml"
            ),
            planning_interval=5,
            step_callbacks=[chain_of_thought_callable],
        )
