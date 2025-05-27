"""
Exploratory Data Analysis (EDA) Agent for data analysis and insights in ML models.

This module defines an EdaAgent that analyzes datasets to generate comprehensive
exploratory data analysis reports before model building begins.
"""

import logging
from typing import List, Callable

from smolagents import LiteLLMModel, CodeAgent

from plexe.config import config, prompt_templates
from plexe.internal.common.utils.agents import get_prompt_templates
from plexe.tools.datasets import register_eda_report, drop_null_columns, get_latest_datasets
from plexe.tools.schemas import get_dataset_schema

logger = logging.getLogger(__name__)


class EdaAgent:
    """
    Agent for performing exploratory data analysis on datasets.

    This agent analyzes the available datasets to produce a comprehensive EDA report
    containing data overview, feature analysis, relationships, data quality issues,
    key insights, and recommendations for modeling.
    """

    def __init__(
        self,
        model_id: str = "openai/gpt-4o",
        verbose: bool = False,
        chain_of_thought_callable: Callable = None,
    ):
        """
        Initialize the EDA agent.

        Args:
            model_id: Model ID for the LLM to use for data analysis
            verbose: Whether to display detailed agent logs
            chain_of_thought_callable: Optional callable for chain of thought logging
        """
        self.model_id = model_id
        self.verbose = verbose

        # Set verbosity level
        self.verbosity = 1 if verbose else 0

        # Create the EDA agent with the necessary tools
        self.agent = CodeAgent(
            name="DatasetAnalyser",
            description=(
                "Expert data analyst that performs exploratory data analysis on datasets to generate insights "
                "and recommendations for ML modeling. Will analyse existing datasets, not create new ones.\n"
                "To work effectively, as part of the 'task' prompt the agent STRICTLY requires:\n"
                "- the ML task definition (i.e. 'intent')\n"
                "- the name of the dataset to be analysed"
            ),
            model=LiteLLMModel(model_id=self.model_id),
            tools=[drop_null_columns, register_eda_report, get_dataset_schema, get_latest_datasets],
            add_base_tools=False,
            verbosity_level=self.verbosity,
            # planning_interval=3,
            max_steps=30,
            step_callbacks=[chain_of_thought_callable],
            additional_authorized_imports=[
                "pandas",
                "pandas.*",
                "numpy",
                "numpy.*",
                "plexe",
                "plexe.*",
                "scipy",
                "scipy.*",
                "sklearn",
                "sklearn.*",
                "statsmodels",
                "statsmodels.*",
            ]
            + config.code_generation.authorized_agent_imports,
            prompt_templates=get_prompt_templates("code_agent.yaml", "eda_prompt_templates.yaml"),
        )

    def run(
        self,
        intent: str,
        dataset_names: List[str],
    ) -> bool:
        """
        Run the EDA agent to analyze datasets and create EDA reports.

        Args:
            intent: Natural language description of the model's purpose
            dataset_names: List of dataset registry names available for analysis

        Returns:
            Dictionary containing:
            - eda_report_names: List of registered EDA report names in the Object Registry
            - dataset_names: List of datasets that were analyzed
            - summary: Brief summary of key findings
        """
        # Use the template system to create the prompt
        datasets_str = ", ".join(dataset_names)

        # Generate the prompt using the template system
        task_description = prompt_templates.eda_agent_prompt(
            intent=intent,
            datasets=datasets_str,
        )

        # Run the agent to get analysis
        self.agent.run(task_description)

        return True
