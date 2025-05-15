"""
Exploratory Data Analysis (EDA) Agent for data analysis and insights in ML models.

This module defines an EdaAgent that analyzes datasets to generate comprehensive
exploratory data analysis reports before model building begins.
"""

import logging
from typing import Dict, List, Any, Callable

from smolagents import LiteLLMModel, CodeAgent

from plexe.config import prompt_templates
from plexe.internal.common.registries.objects import ObjectRegistry
from plexe.internal.common.utils.agents import get_prompt_templates
from plexe.internal.models.tools.datasets import register_eda_report
from plexe.internal.models.tools.schemas import get_raw_dataset_schema

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
                "Expert data analyst that performs exploratory data analysis on datasets "
                "to generate insights and recommendations for ML modeling."
            ),
            model=LiteLLMModel(model_id=self.model_id),
            tools=[register_eda_report, get_raw_dataset_schema],
            add_base_tools=False,
            verbosity_level=self.verbosity,
            planning_interval=3,
            step_callbacks=[chain_of_thought_callable],
            additional_authorized_imports=["pandas", "numpy", "plexe"],
            prompt_templates=get_prompt_templates("code_agent.yaml", "eda_prompt_templates.yaml"),
        )

    def run(
        self,
        intent: str,
        dataset_names: List[str],
    ) -> Dict[str, Any]:
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

        # Get the registered EDA reports from the registry
        object_registry = ObjectRegistry()
        eda_report_names = [name for name in object_registry.list() if str(dict) in name and "eda_report_" in name]

        # Extract report summaries
        summaries = []
        for report_name in eda_report_names:
            # Extract dataset name from report name (remove type prefix and eda_report_ prefix)
            parts = report_name.split("://")
            if len(parts) > 1:
                dataset_name = parts[1].replace("eda_report_", "")
                report = object_registry.get(dict, f"eda_report_{dataset_name}")
                if report and "insights" in report:
                    # Get the first few insights as summary
                    insights = report.get("insights", [])
                    if insights:
                        summaries.extend(insights[:2])  # Take first two insights

        # Return reports and indicate they're already registered
        return {
            "eda_report_names": eda_report_names,
            "dataset_names": dataset_names,
            "summary": summaries[:3],  # Limit to 3 most important insights
        }
