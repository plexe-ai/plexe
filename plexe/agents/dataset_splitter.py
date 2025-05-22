"""
Dataset Splitter Agent for partitioning datasets into training, validation, and test sets.

This module defines a DatasetSplitterAgent that handles the critical task of properly
splitting datasets for machine learning model development, supporting both random and
time-series-aware splitting strategies.
"""

import logging
from typing import Optional, Callable

from smolagents import CodeAgent, LiteLLMModel

from plexe.config import config
from plexe.internal.common.utils.agents import get_prompt_templates
from plexe.tools.datasets import get_dataset_preview, get_eda_report
from plexe.tools.datasets import register_split_datasets

logger = logging.getLogger(__name__)


class DatasetSplitterAgent:
    """
    Agent for intelligently splitting datasets into train, validation, and test sets.

    This agent analyzes datasets and performs appropriate splits based on data characteristics
    and the specific ML task at hand, handling both standard random splits and specialized
    cases like time-series, imbalanced classification, and small datasets.
    """

    def __init__(
        self,
        model_id: str,
        verbose: bool = False,
        chain_of_thought_callable: Optional[Callable] = None,
    ):
        """
        Initialize the dataset splitter agent.

        Args:
            model_id: Model ID for the LLM to use for split decision-making
            verbose: Whether to display detailed agent logs
            chain_of_thought_callable: Optional callable for chain of thought logging
        """
        # Set verbosity level
        self.verbosity = 1 if verbose else 0

        # Create the dataset splitter agent with the necessary tools
        self.agent = CodeAgent(
            name="DatasetSplitter",
            description=(
                "Expert data engineer that intelligently splits datasets for machine learning tasks. "
                "To work effectively, as part of the 'task' prompt the agent STRICTLY requires:"
                "- the ML task definition (i.e. 'intent')"
                "- the registered NAME of the dataset to split"
                "- the split ratios (train_ratio, val_ratio, test_ratio)"
                "- any helpful information or specific requirements for the split"
            ),
            model=LiteLLMModel(model_id=model_id),
            tools=[
                get_dataset_preview,
                get_eda_report,
                register_split_datasets,
            ],
            planning_interval=5,
            add_base_tools=False,
            verbosity_level=self.verbosity,
            additional_authorized_imports=[
                "pandas",
                "pandas.*",
                "numpy",
                "numpy.*",
                "plexe",
                "plexe.*",
                "sklearn",
                "sklearn.*",
                "sklearn.model_selection",
                "sklearn.model_selection.*",
                "scipy",
                "scipy.*",
                "datetime",
                "datetime.*",
            ]
            + config.code_generation.authorized_agent_imports,
            prompt_templates=get_prompt_templates(
                base_template_name="code_agent.yaml", override_template_name="dataset_splitter_templates.yaml"
            ),
            step_callbacks=[chain_of_thought_callable],
        )
