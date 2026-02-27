"""
ML Task Analyser Agent.

Analyzes dataset in ML context and determines task type, target variable, etc.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from plexe.utils.litellm_wrapper import PlexeLiteLLMModel
from smolagents import CodeAgent

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

from plexe.models import BuildContext
from plexe.config import Config
from plexe.tools.submission import get_register_eda_report_tool
from plexe.utils.tracing import agent_span
from plexe.config import get_routing_for_model

logger = logging.getLogger(__name__)


class MLTaskAnalyserAgent:
    """
    Agent that analyzes dataset from ML perspective.

    Narrow responsibility: Determine task type, target variable, ML-specific insights.
    """

    def __init__(
        self,
        spark: SparkSession,
        dataset_uri: str,
        stats_report: dict,
        context: BuildContext,
        config: Config,
    ):
        self.spark = spark
        self.dataset_uri = dataset_uri
        self.stats_report = stats_report
        self.context = context
        self.config = config
        self.llm_model = config.ml_task_analysis_llm

    def _build_agent(self) -> CodeAgent:
        """Build CodeAgent with structured submission tool."""
        # Check for user feedback
        from plexe.agents.utils import format_user_feedback_for_prompt

        feedback_section = format_user_feedback_for_prompt(self.context.scratch.get("_user_feedback"))

        # Get routing configuration for this agent's model
        api_base, headers = get_routing_for_model(self.config.routing_config, self.llm_model)

        return CodeAgent(
            name="MLTaskAnalyser",
            instructions=(
                "## YOUR ROLE:\n"
                "Perform ML-focused data analysis to guide feature engineering and modeling. Your work is "
                "in support of a broader ML task, which will be referred to as the 'ML task'.\n"
                "\n"
                f"{feedback_section}"
                "## INPUTS PROVIDED:\n"
                "Your PySpark code has access to these variables:\n"
                "- `spark`: PySpark SparkSession (use interactively)\n"
                "- `dataset_uri`: Path to dataset (parquet format)\n"
                "- `stats_report`: Statistical profiling from previous analysis\n"
                "\n"
                "## YOUR FOCUS (ML-Specific Insights):\n"
                "Build on statistical foundation and analyze:\n"
                "1. **Task Type**: Classification (binary/multiclass), regression, etc.\n"
                "2. **Target Variable**: Which column to predict - examine distribution, cardinality\n"
                "3. **Relationships with Target**: For numeric features, compute correlations; for other column types, note observable patterns\n"
                "4. **ML Quality Issues**: Class imbalance, data leakage, outliers, suspicious patterns\n"
                "5. **Data Challenges**: Issues specific to this data type (e.g., path validity, text length variance, missing values)\n"
                "6. **Preprocessing Recommendations**: High-level guidance on preparing data for modeling\n"
                "7. **Feature Engineering** (if applicable): Suggested transformations, interactions, encodings\n"
                "8. **Split Strategy** (in recommended_split):\n"
                "   - temporal_reasoning: Chronological split only if predicting FUTURE time periods. Timestamp as metadata → random OK.\n"
                "   - stratification_reasoning: Stratify if classification with class imbalance.\n"
                "\n"
                "## HOW TO ANALYZE:\n"
                "Write PySpark code directly:\n"
                "  df = spark.read.parquet(dataset_uri)\n"
                "  # Identify target column by examining cardinality/distributions\n"
                "  # Compute correlations (numeric features only!)\n"
                "  # Analyze class balance if classification\n"
                "\n"
                "## OUTPUT:\n"
                "When analysis complete, call save_eda_report(...) with ALL required parameters.\n"
                "The function signature documents what's needed.\n"
                "\n"
                "## RULES:\n"
                "- ANALYSIS ONLY - do not transform datasets (done by downstream agents)\n"
                "- Focus on ACTIONABLE insights that inform feature engineering and modeling\n"
                "- Incorporate statistical profile findings into your analysis\n"
            ),
            model=PlexeLiteLLMModel(
                model_id=self.llm_model,
                temperature=self.config.get_temperature("ml_task_analyser"),
                api_base=api_base,
                extra_headers=headers,
            ),
            verbosity_level=self.config.agent_verbosity_level,
            tools=[get_register_eda_report_tool(self.context)],
            add_base_tools=False,
            additional_authorized_imports=self.config.allowed_base_imports
            + ["pyspark", "pyspark.*", "pyspark.sql", "pyspark.sql.*", "pyspark.sql.functions"],
            max_steps=25,
        )

    @agent_span("MLTaskAnalyserAgent")
    def run(self) -> dict:
        """Run ML task analysis."""

        logger.info("Starting ML task analysis...")

        agent = self._build_agent()

        task = (
            f"Analyze the dataset from an ML perspective for this task:\n\n"
            f"**ML TASK**: {self.context.intent}\n\n"
            f"Statistical profiling (already completed) is available in `stats_report` variable.\n\n"
            f"Determine the ML task type and target variable(s) that align with the broader ML task. "
            f"Provide actionable modeling insights."
        )

        agent.run(
            task=task,
            additional_args={"spark": self.spark, "dataset_uri": self.dataset_uri, "stats_report": self.stats_report},
        )

        report = self.context.scratch.get("_eda_report")

        if not report:
            logger.warning("Agent completed but no report was saved")
            return {"error": "No report generated"}

        # Extract and save output_targets to context
        output_targets = report.get("output_targets", [])
        if not output_targets:
            raise ValueError("MLTaskAnalyser failed to identify output_targets")

        self.context.output_targets = output_targets

        logger.info(f"ML task analysis completed: {report.get('task_type', 'unknown')} with targets {output_targets}")
        return report
